//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements affine fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/IRMapping.h"
#include <iomanip>
#include <optional>
#include <sstream>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPFUSION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-loop-fusion"

using namespace mlir;
using namespace mlir::affine;

namespace {
/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.
// TODO: Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
struct LoopFusion : public affine::impl::AffineLoopFusionBase<LoopFusion> {
  LoopFusion() = default;
  LoopFusion(unsigned fastMemorySpace, uint64_t localBufSizeThresholdBytes,
             bool maximalFusion, enum FusionMode affineFusionMode) {
    this->fastMemorySpace = fastMemorySpace;
    this->localBufSizeThreshold = localBufSizeThresholdBytes / 1024;
    this->maximalFusion = maximalFusion;
    this->affineFusionMode = affineFusionMode;
  }

  void runOnBlock(Block *block);
  void runOnOperation() override;
};

} // namespace

/// Returns true if node 'srcId' can be removed after fusing it with node
/// 'dstId'. The node can be removed if any of the following conditions are met:
///   1. 'srcId' has no output dependences after fusion and no escaping memrefs.
///   2. 'srcId' has no output dependences after fusion, has escaping memrefs
///       and the fusion slice is maximal.
///   3. 'srcId' has output dependences after fusion, the fusion slice is
///      maximal and the fusion insertion point dominates all the dependences.
static bool canRemoveSrcNodeAfterFusion(
    unsigned srcId, unsigned dstId, const ComputationSliceState &fusionSlice,
    Operation *fusedLoopInsPoint, const DenseSet<Value> &escapingMemRefs,
    MemRefDependenceGraph *mdg) {

  Operation *dstNodeOp = mdg->getNode(dstId)->op;
  bool hasOutDepsAfterFusion = false;

  for (auto &outEdge : mdg->outEdges[srcId]) {
    Operation *depNodeOp = mdg->getNode(outEdge.id)->op;
    // Skip dependence with dstOp since it will be removed after fusion.
    if (depNodeOp == dstNodeOp)
      continue;

    // Only fusion within the same block is supported. Use domination analysis
    // when needed.
    if (depNodeOp->getBlock() != dstNodeOp->getBlock())
      return false;

    // Check if the insertion point of the fused loop dominates the dependence.
    // Otherwise, the src loop can't be removed.
    if (fusedLoopInsPoint != depNodeOp &&
        !fusedLoopInsPoint->isBeforeInBlock(depNodeOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: dst loop doesn't "
                                 "dominate dependence\n");
      return false;
    }

    hasOutDepsAfterFusion = true;
  }

  // If src loop has dependences after fusion or it writes to an live-out or
  // escaping memref, we can only remove it if the fusion slice is maximal so
  // that all the dependences are preserved.
  if (hasOutDepsAfterFusion || !escapingMemRefs.empty()) {
    std::optional<bool> isMaximal = fusionSlice.isMaximal();
    if (!isMaximal) {
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: can't determine "
                                 "if fusion is maximal\n");
      return false;
    }

    if (!*isMaximal) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Src loop can't be removed: fusion is not maximal\n");
      return false;
    }
  }

  return true;
}

/// Returns in 'srcIdCandidates' the producer fusion candidates for consumer
/// 'dstId'. Candidates are sorted by node id order. This order corresponds to
/// the program order when the 'mdg' is created. However, program order is not
/// guaranteed and must not be required by the client. Program order won't be
/// held if the 'mdg' is reused from a previous fusion step or if the node
/// creation order changes in the future to support more advance cases.
// TODO: Move this to a loop fusion utility once 'mdg' is also moved.
static void getProducerCandidates(unsigned dstId, MemRefDependenceGraph *mdg,
                                  SmallVectorImpl<unsigned> &srcIdCandidates) {
  // Skip if no input edges along which to fuse.
  if (mdg->inEdges.count(dstId) == 0)
    return;

  // Gather memrefs from loads in 'dstId'.
  auto *dstNode = mdg->getNode(dstId);
  DenseSet<Value> consumedMemrefs;
  for (Operation *load : dstNode->loads)
    consumedMemrefs.insert(cast<AffineReadOpInterface>(load).getMemRef());

  // Traverse 'dstId' incoming edges and gather the nodes that contain a store
  // to one of the consumed memrefs.
  for (auto &srcEdge : mdg->inEdges[dstId]) {
    auto *srcNode = mdg->getNode(srcEdge.id);
    // Skip if 'srcNode' is not a loop nest.
    if (!isa<AffineForOp>(srcNode->op))
      continue;

    if (any_of(srcNode->stores, [&](Operation *op) {
          auto storeOp = cast<AffineWriteOpInterface>(op);
          return consumedMemrefs.count(storeOp.getMemRef()) > 0;
        }))
      srcIdCandidates.push_back(srcNode->id);
  }

  llvm::sort(srcIdCandidates);
  srcIdCandidates.erase(llvm::unique(srcIdCandidates), srcIdCandidates.end());
}

// modifiy py p

/// 辅助函数：检查操作是否在给定范围内
static bool isOpInRange(Operation *op, Operation *rangeStart, Operation *rangeEnd) {
  // 检查op是否在rangeStart和rangeEnd之间（包括它们的内部）
  
  // 首先检查是否在同一个block中
  if (op->getBlock() != rangeStart->getBlock() || 
      rangeStart->getBlock() != rangeEnd->getBlock()) {
    return false;
  }
  
  // 检查是否在操作内部（遍历操作树）
  // 使用walk来检查是否是子操作
  bool foundInRangeStart = false, foundInRangeEnd = false;
  
  rangeStart->walk([&](Operation *subOp) {
    if (subOp == op) {
      foundInRangeStart = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (foundInRangeStart)
    return true;
    
  rangeEnd->walk([&](Operation *subOp) {
    if (subOp == op) {
      foundInRangeEnd = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  if (foundInRangeEnd)
    return true;
  
  // 检查是否在两个操作之间
  if (rangeStart->isBeforeInBlock(op) && op->isBeforeInBlock(rangeEnd)) {
    return true;
  }
  
  return false;
}

/// 检查reinterpret_cast操作是否会在指定范围内被使用
/// 如果reinterpret_cast的结果没有在srcOp到dstOp之间被使用，则认为不影响融合
static bool reinterpretCastUsedInFusionRange(memref::ReinterpretCastOp reinterpretOp,
                                           Operation *srcOp, Operation *dstOp) {
  Value result = reinterpretOp.getResult();
  
  // 检查reinterpret_cast的结果是否有任何使用
  if (result.use_empty()) {
    LLVM_DEBUG(llvm::dbgs() << "ReinterpretCast result is unused, safe for fusion\n");
    return false;
  }
  
  // 检查所有使用点是否在融合范围内
  for (auto &use : result.getUses()) {
    Operation *user = use.getOwner();
    
    // 检查使用点是否在srcOp和dstOp之间（包括它们内部）
    // 如果在这个范围内，则可能影响融合安全性
    if (isOpInRange(user, srcOp, dstOp)) {
      LLVM_DEBUG(llvm::dbgs() << "ReinterpretCast result used within fusion range: " 
                              << *user << "\n");
      return true;
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "ReinterpretCast result not used within fusion range, safe for fusion\n");
  return false;
}

// static bool areMemrefsAliased(Value memref1, Value memref2) {
//   // 直接相等
//   if (memref1 == memref2)
//     return true;
    
//   // 检查是否通过reinterpret_cast相关
//   auto checkReinterpretCastAlias = [](Value src, Value dst) -> bool {
//     auto reinterpretOp = src.getDefiningOp<memref::ReinterpretCastOp>();
//     if (!reinterpretOp)
//       return false;
      
//     auto srcType = reinterpretOp.getSource().getType().cast<MemRefType>();
//     auto dstType = reinterpretOp.getResult().getType().cast<MemRefType>();
    
//     // 必须是减少一个维度的reshape（4D -> 3D）
//     if (srcType.getRank() != dstType.getRank() + 1 || srcType.getRank() < 2)
//       return false;
      
//     // 验证stride是否连续
//     auto reinterpretStrides = reinterpretOp.getStaticStrides();
//     if (reinterpretStrides.empty())
//       return false;
      
//     // 检查source memref是否指向正确的目标
//     if (reinterpretOp.getSource() != dst)
//       return false;
    
//     // 模式1: 前两个维度flatten (如 16x32x28x28 -> 512x28x28)
//     auto checkFrontFlatten = [&]() -> bool {
//       if (srcType.getRank() < 2 || dstType.getRank() < 1)
//         return false;
        
//       auto srcDim0 = srcType.getDimSize(0);
//       auto srcDim1 = srcType.getDimSize(1);
//       auto dstDim0 = dstType.getDimSize(0);
      
//       if (srcDim0 == ShapedType::kDynamic || 
//           srcDim1 == ShapedType::kDynamic ||
//           dstDim0 == ShapedType::kDynamic)
//         return false;
        
//       // 验证flatten: srcDim0 * srcDim1 = dstDim0
//       if (srcDim0 * srcDim1 != dstDim0)
//         return false;
        
//       // 验证剩余维度一致
//       for (int i = 1; i < dstType.getRank(); i++) {
//         if (i + 1 >= srcType.getRank() || 
//             srcType.getDimSize(i + 1) != dstType.getDimSize(i))
//           return false;
//       }
      
//       return true;
//     };
    
//     // 模式2: 后两个维度flatten (如 16x64x56x56 -> 16x64x3136)
//     auto checkBackFlatten = [&]() -> bool {
//       if (srcType.getRank() < 2 || dstType.getRank() < 1)
//         return false;
        
//       int srcRank = srcType.getRank();
//       int dstRank = dstType.getRank();
      
//       // 获取要flatten的后两个维度
//       auto srcSecondLast = srcType.getDimSize(srcRank - 2);
//       auto srcLast = srcType.getDimSize(srcRank - 1);
//       auto dstLast = dstType.getDimSize(dstRank - 1);
      
//       if (srcSecondLast == ShapedType::kDynamic || 
//           srcLast == ShapedType::kDynamic ||
//           dstLast == ShapedType::kDynamic)
//         return false;
        
//       // 验证flatten: srcSecondLast * srcLast = dstLast
//       if (srcSecondLast * srcLast != dstLast)
//         return false;
        
//       // 验证前面的维度一致
//       for (int i = 0; i < dstRank - 1; i++) {
//         if (i >= srcRank - 2 || 
//             srcType.getDimSize(i) != dstType.getDimSize(i))
//           return false;
//       }
      
//       return true;
//     };
    
//     return checkFrontFlatten() || checkBackFlatten();
//   };
  
//   // 双向检查reinterpret_cast关系
//   return checkReinterpretCastAlias(memref1, memref2) || 
//          checkReinterpretCastAlias(memref2, memref1);
// }

// /// 获取memref的根源，穿透reinterpret_cast操作
// static Value getRootMemRef(Value memref) {
//   while (auto reinterpretOp = memref.getDefiningOp<memref::ReinterpretCastOp>()) {
//     memref = reinterpretOp.getSource();
//   }
//   return memref;
// }


/// Returns in 'producerConsumerMemrefs' the memrefs involved in a
/// producer-consumer dependence between 'srcId' and 'dstId'.
static void
gatherProducerConsumerMemrefs(unsigned srcId, unsigned dstId,
                              MemRefDependenceGraph *mdg,
                              DenseSet<Value> &producerConsumerMemrefs) {
  auto *dstNode = mdg->getNode(dstId);
  auto *srcNode = mdg->getNode(srcId);
  gatherProducerConsumerMemrefs(srcNode->stores, dstNode->loads,
                                producerConsumerMemrefs);
}

// static void gatherProducerConsumerMemrefs(unsigned srcId, unsigned dstId,
//                                           MemRefDependenceGraph *mdg,
//                                           DenseSet<Value> &producerConsumerMemrefs) {
//   auto *dstNode = mdg->getNode(dstId);
//   auto *srcNode = mdg->getNode(srcId);
  
//   // 原有逻辑
//   gatherProducerConsumerMemrefs(srcNode->stores, dstNode->loads,
//                                 producerConsumerMemrefs);
  
//   // 增强逻辑：检查reinterpret_cast别名
//   for (Operation *storeOp : srcNode->stores) {
//     Value storeMemref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
    
//     for (Operation *loadOp : dstNode->loads) {
//       Value loadMemref = cast<AffineReadOpInterface>(loadOp).getMemRef();
      
//       if (areMemrefsAliased(storeMemref, loadMemref)) {
//         // 添加原始memref和别名memref
//         producerConsumerMemrefs.insert(storeMemref);
//         producerConsumerMemrefs.insert(loadMemref);
        
//         // 也添加根memref以确保完整性
//         producerConsumerMemrefs.insert(getRootMemRef(storeMemref));
//         producerConsumerMemrefs.insert(getRootMemRef(loadMemref));
//       }
//     }
//   }
// }


/// A memref escapes in the context of the fusion pass if either:
///   1. it (or its alias) is a block argument, or
///   2. created by an op not known to guarantee alias freedom,
///   3. it (or its alias) are used by ops other than affine dereferencing ops
///   (e.g., by call op, memref load/store ops, alias creating ops, unknown ops,
///   terminator ops, etc.); such ops do not deference the memref in an affine
///   way.

/// 检查reinterpret_cast操作是否实际上未被使用
/// 这种操作不应该阻止loop fusion
static bool isUnusedReinterpretCast(Operation *op) {
  auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(op);
  if (!reinterpretOp)
    return false;
  
  // 检查reinterpret_cast的结果是否有任何真实使用
  Value result = reinterpretOp.getResult();
  return result.use_empty();
}


// static bool isEscapingMemref(Value memref, Block *block) { // src
// modified py p
static bool isEscapingMemref(Value memref, Block *block, 
                            Operation *srcOp = nullptr, Operation *dstOp = nullptr) {

  Operation *defOp = memref.getDefiningOp();
  // Check if 'memref' is a block argument.
  if (!defOp)
    return true;

  // Check if this is defined to be an alias of another memref.
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(defOp))
    if (isEscapingMemref(viewOp.getViewSource(), block))
      return true;

  // Any op besides allocating ops wouldn't guarantee alias freedom
  if (!hasSingleEffect<mlir::MemoryEffects::Allocate>(defOp, memref))
    return true;

  // Check if 'memref' is used by a non-deferencing op (including unknown ones)
  // (e.g., call ops, alias creating ops, etc.).
  return llvm::any_of(memref.getUsers(), [&](Operation *user) {
    // Ignore users outside of `block`.
    Operation *ancestorOp = block->getParent()->findAncestorOpInRegion(*user);
    if (!ancestorOp)
      return true;
    if (ancestorOp->getBlock() != block)
      return false;

    if (isa<mlir::UnrealizedConversionCastOp>(user)) {
      LLVM_DEBUG(llvm::dbgs() << "Ignoring unrealized_conversion_cast: " << *user << "\n");
      return false;  // 不认为是逃逸使用
    }

    // 新增：特殊处理reinterpret_cast操作
    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(user)) {
      // 如果提供了srcOp和dstOp，进行精细分析
      if (srcOp && dstOp) {
        // 检查reinterpret_cast是否在融合范围内被使用
        if (!reinterpretCastUsedInFusionRange(reinterpretOp, srcOp, dstOp)) {
          LLVM_DEBUG(llvm::dbgs() << "Ignoring safe reinterpret_cast: " << *user << "\n");
          return false;  // 不认为是逃逸使用
        }
      }
      // 如果没有提供范围信息或在范围内被使用，则保持保守策略
      LLVM_DEBUG(llvm::dbgs() << "ReinterpretCast affects fusion safety: " << *user << "\n");
      return true;
    }

    return !isa<AffineMapAccessInterface>(*user);
  });
}

/// Returns in 'escapingMemRefs' the memrefs from affine store ops in node 'id'
/// that escape the block or are accessed in a non-affine way.
static void gatherEscapingMemrefs(unsigned id, MemRefDependenceGraph *mdg,
                                  DenseSet<Value> &escapingMemRefs) {
  auto *node = mdg->getNode(id);
  for (Operation *storeOp : node->stores) {
    auto memref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
    if (escapingMemRefs.count(memref))
      continue;
    if (isEscapingMemref(memref, &mdg->block))
      escapingMemRefs.insert(memref);
  }
}

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
// This can increase the loop depth at which we can fuse a slice, since we are
// pushing loop carried dependence to a greater depth in the loop nest.
static void sinkSequentialLoops(MemRefDependenceGraph::Node *node) {
  assert(isa<AffineForOp>(node->op));
  AffineForOp newRootForOp = sinkSequentialLoops(cast<AffineForOp>(node->op));
  node->op = newRootForOp;
}

// Creates and returns a private (single-user) memref for fused loop rooted
// at 'forOp', with (potentially reduced) memref size based on the
// MemRefRegion written to by 'srcStoreOpInst' at depth 'dstLoopDepth'.
// TODO: consider refactoring the common code from generateDma and
// this one.
static Value createPrivateMemRef(AffineForOp forOp, Operation *srcStoreOpInst,
                                 unsigned dstLoopDepth,
                                 std::optional<unsigned> fastMemorySpace,
                                 uint64_t localBufSizeThreshold) {
  Operation *forInst = forOp.getOperation();

  // Create builder to insert alloc op just before 'forOp'.
  OpBuilder b(forInst);
  // Builder to create constants at the top level.
  OpBuilder top(forInst->getParentRegion());
  // Create new memref type based on slice bounds.
  auto oldMemRef = cast<AffineWriteOpInterface>(srcStoreOpInst).getMemRef();
  auto oldMemRefType = cast<MemRefType>(oldMemRef.getType());
  unsigned rank = oldMemRefType.getRank();

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region(srcStoreOpInst->getLoc());
  bool validRegion = succeeded(region.compute(srcStoreOpInst, dstLoopDepth));
  (void)validRegion;
  assert(validRegion && "unexpected memref region failure");
  SmallVector<int64_t, 4> newShape;
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  // Query 'region' for 'newShape' and lower bounds of MemRefRegion accessed
  // by 'srcStoreOpInst' at depth 'dstLoopDepth'.
  std::optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&newShape, &lbs, &lbDivisors);
  assert(numElements && "non-constant number of elts in local buffer");

  const FlatAffineValueConstraints *cst = region.getConstraints();
  // 'outerIVs' holds the values that this memory region is symbolic/parametric
  // on; this would correspond to loop IVs surrounding the level at which the
  // slice is being materialized.
  SmallVector<Value, 8> outerIVs;
  cst->getValues(rank, cst->getNumVars(), &outerIVs);

  // Build 'rank' AffineExprs from MemRefRegion 'lbs'
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);
  for (unsigned d = 0; d < rank; ++d) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++) {
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    }
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);
    offsets.push_back(offset);
  }

  // Create 'newMemRefType' using 'newShape' from MemRefRegion accessed
  // by 'srcStoreOpInst'.
  auto eltSize = getMemRefIntOrFloatEltSizeInBytes(oldMemRefType);
  assert(eltSize && "memrefs with size elt types expected");
  uint64_t bufSize = *eltSize * *numElements;
  unsigned newMemSpace;
  if (bufSize <= localBufSizeThreshold && fastMemorySpace.has_value()) {
    newMemSpace = *fastMemorySpace;
  } else {
    newMemSpace = oldMemRefType.getMemorySpaceAsInt();
  }
  auto newMemRefType = MemRefType::get(newShape, oldMemRefType.getElementType(),
                                       {}, newMemSpace);

  // Create new private memref for fused loop 'forOp'. 'newShape' is always
  // a constant shape.
  // TODO: Create/move alloc ops for private memrefs closer to their
  // consumer loop nests to reduce their live range. Currently they are added
  // at the beginning of the block, because loop nests can be reordered
  // during the fusion pass.
  Value newMemRef = top.create<memref::AllocOp>(forOp.getLoc(), newMemRefType);

  // Build an AffineMap to remap access functions based on lower bound offsets.
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    auto dimExpr = b.getAffineDimExpr(outerIVs.size() + i);

    auto remapExpr =
        simplifyAffineExpr(dimExpr - offsets[i], outerIVs.size() + rank, 0);
    remapExprs.push_back(remapExpr);
  }

  auto indexRemap =
      AffineMap::get(outerIVs.size() + rank, 0, remapExprs, forOp.getContext());

  // Replace all users of 'oldMemRef' with 'newMemRef'.
  LogicalResult res =
      replaceAllMemRefUsesWith(oldMemRef, newMemRef, {}, indexRemap,
                               /*extraOperands=*/outerIVs,
                               /*symbolOperands=*/{},
                               /*domOpFilter=*/&*forOp.getBody()->begin());
  assert(succeeded(res) &&
         "replaceAllMemrefUsesWith should always succeed here");
  (void)res;
  return newMemRef;
}

/// Walking from node 'srcId' to node 'dstId' (exclusive of 'srcId' and
/// 'dstId'), if there is any non-affine operation accessing 'memref', return
/// true. Otherwise, return false.
static bool hasNonAffineUsersOnThePath(unsigned srcId, unsigned dstId,
                                       Value memref,
                                       MemRefDependenceGraph *mdg) {
  auto *srcNode = mdg->getNode(srcId);
  auto *dstNode = mdg->getNode(dstId);
  Value::user_range users = memref.getUsers();
  // For each MemRefDependenceGraph's node that is between 'srcNode' and
  // 'dstNode' (exclusive of 'srcNodes' and 'dstNode'), check whether any
  // non-affine operation in the node accesses the 'memref'.
  for (auto &idAndNode : mdg->nodes) {
    Operation *op = idAndNode.second.op;
    // Take care of operations between 'srcNode' and 'dstNode'.
    if (srcNode->op->isBeforeInBlock(op) && op->isBeforeInBlock(dstNode->op)) {
      // Walk inside the operation to find any use of the memref.
      // Interrupt the walk if found.
      auto walkResult = op->walk([&](Operation *user) {
        // Skip affine ops.
        if (isa<AffineMapAccessInterface>(*user))
          return WalkResult::advance();

        // 新增：智能处理reinterpret_cast操作
        if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(user)) {
          if (llvm::is_contained(users, user)) {
            // 检查这个reinterpret_cast是否真的会影响融合
            if (!reinterpretCastUsedInFusionRange(reinterpretOp, srcNode->op, dstNode->op)) {
              LLVM_DEBUG(llvm::dbgs() << "Skipping safe reinterpret_cast on path: " 
                                      << *user << "\n");
              return WalkResult::advance();
            }
          }
        }

        // Find a non-affine op that uses the memref.
        if (llvm::is_contained(users, user))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return true;
    }
  }
  return false;
}

/// Check whether a memref value in node 'srcId' has a non-affine that
/// is between node 'srcId' and node 'dstId' (exclusive of 'srcNode' and
/// 'dstNode').
static bool hasNonAffineUsersOnThePath(unsigned srcId, unsigned dstId,
                                       MemRefDependenceGraph *mdg) {
  // Collect memref values in node 'srcId'.
  auto *srcNode = mdg->getNode(srcId);
  llvm::SmallDenseSet<Value, 2> memRefValues;
  srcNode->op->walk([&](Operation *op) {
    // Skip affine ops.
    if (isa<AffineForOp>(op))
      return WalkResult::advance();
    for (Value v : op->getOperands())
      // Collect memref values only.
      if (isa<MemRefType>(v.getType()))
        memRefValues.insert(v);
    return WalkResult::advance();
  });
  // Looking for users between node 'srcId' and node 'dstId'.
  return llvm::any_of(memRefValues, [&](Value memref) {
    return hasNonAffineUsersOnThePath(srcId, dstId, memref, mdg);
  });
}

// Checks the profitability of fusing a backwards slice of the loop nest
// surrounding 'srcOpInst' into the loop nest surrounding 'dstLoadOpInsts'.
// The argument 'srcStoreOpInst' is used to calculate the storage reduction on
// the memref being produced and consumed, which is an input to the cost model.
// For producer-consumer fusion, 'srcStoreOpInst' will be the same as
// 'srcOpInst', as we are slicing w.r.t to that producer. For input-reuse
// fusion, 'srcOpInst' will be the src loop nest LoadOp which reads from the
// same memref as dst loop nest load ops, and 'srcStoreOpInst' will be the
// unique store op in the src node, which will be used to check that the write
// region is the same after input-reuse fusion. Computation slices are provided
// in 'depthSliceUnions' for each legal fusion depth. The maximal depth at which
// fusion is legal is provided in 'maxLegalFusionDepth'. Returns true if it is
// profitable to fuse the candidate loop nests. Returns false otherwise.
// `dstLoopDepth` is set to the most profitable depth at which to materialize
// the source loop nest slice.
// The profitability model executes the following steps:
// *) Computes the backward computation slice at 'srcOpInst'. This
//    computation slice of the loop nest surrounding 'srcOpInst' is
//    represented by modified src loop bounds in 'sliceState', which are
//    functions of loop IVs in the loop nest surrounding 'srcOpInst'.
// *) Computes the cost of unfused src/dst loop nests (currently the cost of a
//    loop nest is the total number of dynamic operation instances in the loop
//    nest).
// *) Computes the cost of fusing a slice of the src loop nest into the dst
//    loop nest at various values of dst loop depth, attempting to fuse
//    the largest computation slice at the maximal dst loop depth (closest to
//    the load) to minimize reuse distance and potentially enable subsequent
//    load/store forwarding.
//    NOTE: 'dstLoopDepth' refers to the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    excessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
// TODO: Extend profitability analysis to support scenarios with multiple
// stores.
static bool isFusionProfitable(Operation *srcOpInst, Operation *srcStoreOpInst,
                               AffineForOp dstForOp,
                               ArrayRef<ComputationSliceState> depthSliceUnions,
                               unsigned maxLegalFusionDepth,
                               unsigned *dstLoopDepth,
                               double computeToleranceThreshold) {
  LLVM_DEBUG({
    llvm::dbgs() << "Checking whether fusion is profitable between src op:\n";
    llvm::dbgs() << ' ' << *srcOpInst << " and destination loop:\n";
    llvm::dbgs() << dstForOp << "\n";
  });

  if (maxLegalFusionDepth == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Can't fuse: maxLegalFusionDepth is 0\n");
    return false;
  }

  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getAffineForIVs(*srcOpInst, &srcLoopIVs);

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcLoopIVs[0], &srcLoopNestStats))
    return false;

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats))
    return false;

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxLegalLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  uint64_t minFusedLoopNestComputeCost = std::numeric_limits<uint64_t>::max();
  double maxStorageReduction = 0.0;
  std::optional<uint64_t> sliceMemEstimate;

  // The best loop depth at which to materialize the slice.
  std::optional<unsigned> bestDstLoopDepth;

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcLoopIVs[0], srcLoopNestStats);

  // Compute src loop nest write region size.
  MemRefRegion srcWriteRegion(srcStoreOpInst->getLoc());
  if (failed(srcWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for source operation\n");
    return false;
  }

  std::optional<int64_t> maybeSrcWriteRegionSizeBytes =
      srcWriteRegion.getRegionSize();
  if (!maybeSrcWriteRegionSizeBytes.has_value())
    return false;
  int64_t srcWriteRegionSizeBytes = *maybeSrcWriteRegionSizeBytes;

  // Compute op instance count for the src loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats);

  // Evaluate all depth choices for materializing the slice in the destination
  // loop nest.
  for (unsigned i = maxLegalFusionDepth; i >= 1; --i) {
    const ComputationSliceState &slice = depthSliceUnions[i - 1];
    // Skip slice union if it wasn't computed for this depth.
    if (slice.isEmpty())
      continue;

    int64_t fusedLoopNestComputeCost;
    if (!getFusionComputeCost(srcLoopIVs[0], srcLoopNestStats, dstForOp,
                              dstLoopNestStats, slice,
                              &fusedLoopNestComputeCost)) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to compute fusion compute cost\n");
      continue;
    }

    double additionalComputeFraction =
        fusedLoopNestComputeCost /
            (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
        1;

    // Determine what the slice write MemRefRegion would be, if the src loop
    // nest slice 'slice' were to be inserted into the dst loop nest at loop
    // depth 'i'.
    MemRefRegion sliceWriteRegion(srcStoreOpInst->getLoc());
    if (failed(sliceWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0,
                                        &slice))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to compute slice write region at loopDepth: " << i
                 << "\n");
      continue;
    }

    std::optional<int64_t> maybeSliceWriteRegionSizeBytes =
        sliceWriteRegion.getRegionSize();
    if (!maybeSliceWriteRegionSizeBytes.has_value() ||
        *maybeSliceWriteRegionSizeBytes == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get slice write region size at loopDepth: " << i
                 << "\n");
      continue;
    }
    int64_t sliceWriteRegionSizeBytes = *maybeSliceWriteRegionSizeBytes;

    // If we are fusing for reuse, check that write regions remain the same.
    // TODO: Write region check should check sizes and offsets in
    // each dimension, so that we are sure they are covering the same memref
    // region. Also, move this out to a isMemRefRegionSuperSet helper function.
    if (srcOpInst != srcStoreOpInst &&
        sliceWriteRegionSizeBytes != srcWriteRegionSizeBytes)
      continue;

    double storageReduction = static_cast<double>(srcWriteRegionSizeBytes) /
                              static_cast<double>(sliceWriteRegionSizeBytes);

    LLVM_DEBUG({
      std::stringstream msg;
      msg << "  evaluating fusion profitability at depth : " << i << "\n"
          << std::fixed << std::setprecision(2)
          << "   additional compute fraction: "
          << 100.0 * additionalComputeFraction << "%\n"
          << "   storage reduction factor: " << storageReduction << "x\n"
          << "   fused nest cost: " << fusedLoopNestComputeCost << "\n"
          << "   src write region size: " << srcWriteRegionSizeBytes << "\n"
          << "   slice write region size: " << sliceWriteRegionSizeBytes
          << "\n";
      llvm::dbgs() << msg.str();
    });

    // TODO: This is a placeholder cost model.
    // Among all choices that add an acceptable amount of redundant computation
    // (as per computeToleranceThreshold), we will simply pick the one that
    // reduces the intermediary size the most.
    if ((storageReduction > maxStorageReduction) &&
        (additionalComputeFraction < computeToleranceThreshold)) {
      maxStorageReduction = storageReduction;
      bestDstLoopDepth = i;
      minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
      sliceMemEstimate = sliceWriteRegionSizeBytes;
    }
  }

  // A simple cost model: fuse if it reduces the memory footprint.

  if (!bestDstLoopDepth) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "All fusion choices involve more than the threshold amount of "
           "redundant computation; NOT fusing.\n");
    return false;
  }

  if (!bestDstLoopDepth) {
    LLVM_DEBUG(llvm::dbgs() << "no fusion depth could be evaluated.\n");
    return false;
  }

  // Set dstLoopDepth based on best values from search.
  *dstLoopDepth = *bestDstLoopDepth;

  LLVM_DEBUG(
      llvm::dbgs() << " LoopFusion fusion stats:"
                   << "\n  best loop depth: " << bestDstLoopDepth
                   << "\n  src loop nest compute cost: " << srcLoopNestCost
                   << "\n  dst loop nest compute cost: " << dstLoopNestCost
                   << "\n  fused loop nest compute cost: "
                   << minFusedLoopNestComputeCost << "\n");

  auto dstMemSize = getMemoryFootprintBytes(dstForOp);
  auto srcMemSize = getMemoryFootprintBytes(srcLoopIVs[0]);

  std::optional<double> storageReduction;

  if (!dstMemSize || !srcMemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << "  fusion memory benefit cannot be evaluated; NOT fusing.\n");
    return false;
  }

  auto srcMemSizeVal = *srcMemSize;
  auto dstMemSizeVal = *dstMemSize;

  assert(sliceMemEstimate && "expected value");
  auto fusedMem = dstMemSizeVal + *sliceMemEstimate;

  LLVM_DEBUG(llvm::dbgs() << "   src mem: " << srcMemSizeVal << "\n"
                          << "   dst mem: " << dstMemSizeVal << "\n"
                          << "   fused mem: " << fusedMem << "\n"
                          << "   slice mem: " << sliceMemEstimate << "\n");

  if (static_cast<long>(fusedMem) > srcMemSizeVal + dstMemSizeVal) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion is not profitable; NOT fusing.\n");
    return false;
  }
  storageReduction =
      100.0 *
      (1.0 - fusedMem / (static_cast<double>(srcMemSizeVal) + dstMemSizeVal));

  double additionalComputeFraction =
      100.0 * (minFusedLoopNestComputeCost /
                   (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
               1);
  (void)additionalComputeFraction;
  LLVM_DEBUG({
    std::stringstream msg;
    msg << " fusion is most profitable at depth " << *dstLoopDepth << " with "
        << std::setprecision(2) << additionalComputeFraction
        << "% redundant computation and a ";
    msg << (storageReduction ? std::to_string(*storageReduction) : "<unknown>");
    msg << "% storage reduction.\n";
    llvm::dbgs() << msg.str();
  });

  return true;
}

namespace {

// GreedyFusion greedily fuses loop nests which have a producer/consumer or
// input-reuse relationship on a memref, with the goal of improving locality.
//
// The steps of the producer-consumer fusion algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop an AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Look up dependent loop nests which have a single store op to the same
//         memref.
//      *) Check if dependences would be violated by the fusion.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         at a loop depth determined by the cost model in 'isFusionProfitable'.
//      *) Add the newly fused load/store operations to the state,
//         and also add newly fused load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// The steps of the input-reuse fusion algorithm are as follows:
//
// *) Initialize 'worklist' with node ids from the dependence graph.
// *) For each 'dstNode' in the worklist:
//   *) Find a candidate sibling node 'sibNode' to fuse with 'dstNode' which
//      loads from the same memref, but which has no dependence paths to/from.
//   *) Get a computation slice of 'sibLoopNest', which adjusts its loop
//      bounds to be functions of 'dstLoopNest' IVs and symbols.
//   *) Fuse the 'sibLoopNest' computation slice into the 'dstLoopNest',
//      at a loop depth determined by the cost model in 'isFusionProfitable'.
//      This function also checks that the memref write region of 'sibLoopNest',
//      is preserved in the fused loop nest.
//   *) Update graph state to reflect the fusion of 'sibNode' into 'dstNode'.
//
// Given a graph where top-level operations are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO: to fix
// this.
//
// TODO: Experiment with other fusion policies.
struct GreedyFusion {
public:
  // The data dependence graph to traverse during fusion.
  MemRefDependenceGraph *mdg;
  // Worklist of graph nodes visited during the fusion pass.
  SmallVector<unsigned, 8> worklist;
  // Parameter for local buffer size threshold.
  unsigned localBufSizeThreshold;
  // Parameter for fast memory space.
  std::optional<unsigned> fastMemorySpace;
  // If true, ignore any additional (redundant) computation tolerance threshold
  // that would have prevented fusion.
  bool maximalFusion;
  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  double computeToleranceThreshold;

  using Node = MemRefDependenceGraph::Node;

  GreedyFusion(MemRefDependenceGraph *mdg, unsigned localBufSizeThreshold,
               std::optional<unsigned> fastMemorySpace, bool maximalFusion,
               double computeToleranceThreshold)
      : mdg(mdg), localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace), maximalFusion(maximalFusion),
        computeToleranceThreshold(computeToleranceThreshold) {}

  /// Initializes 'worklist' with nodes from 'mdg'.
  void init() {
    // TODO: Add a priority queue for prioritizing nodes by different
    // metrics (e.g. arithmetic intensity/flops-to-bytes ratio).
    worklist.clear();
    for (auto &idAndNode : mdg->nodes) {
      const Node &node = idAndNode.second;
      worklist.push_back(node.id);
    }
  }
  /// Run only sibling fusion on the `mdg`.
  void runSiblingFusionOnly() {
    fuseSiblingNodes();
    eraseUnusedMemRefAllocations();
  }

  /// Run only producer/consumer fusion on the `mdg`.
  void runProducerConsumerFusionOnly() {
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  // Run the GreedyFusion pass.
  // *) First pass through the nodes fuses single-use producer nodes into their
  //    unique consumer.
  // *) Second pass fuses sibling nodes which share no dependence edges.
  // *) Third pass fuses any remaining producer nodes into their users.
  void runGreedyFusion() {
    // TODO: Run this repeatedly until a fixed-point is reached.
    fuseProducerConsumerNodes(/*maxSrcUserCount=*/1);
    fuseSiblingNodes();
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  /// Returns true if a private memref can be created for `memref` given
  /// the fusion scenario reflected by the other arguments.
  bool canCreatePrivateMemRef(Value memref,
                              const DenseSet<Value> &srcEscapingMemRefs,
                              unsigned producerId, unsigned consumerId,
                              bool removeSrcNode) {
    const Node *consumerNode = mdg->getNode(consumerId);
    // If `memref` is an escaping one, do not create a private memref
    // for the below scenarios, since doing so will leave the escaping
    // memref unmodified as all the writes originally meant for the
    // escaping memref would be performed on the private memref:
    // 1. The source is to be removed after fusion,
    // OR
    // 2. The destination writes to `memref`.
    if (srcEscapingMemRefs.count(memref) > 0 &&
        (removeSrcNode || consumerNode->getStoreOpCount(memref) > 0))
      return false;

    // Don't create a private memref if 'srcNode' has in edges on
    // 'memref' or 'dstNode' has out edges on 'memref'.
    if (mdg->getIncomingMemRefAccesses(producerId, memref) > 0 ||
        mdg->getOutEdgeCount(consumerId, memref) > 0)
      return false;

    // If 'srcNode' will be removed but it has out edges on 'memref' to
    // nodes other than 'dstNode', we have to preserve dependences and
    // cannot create a private memref.
    if (removeSrcNode &&
        any_of(mdg->outEdges[producerId], [&](const auto &edge) {
          return edge.value == memref && edge.id != consumerId;
        }))
      return false;

    return true;
  }

  // 在GreedyFusion类中添加的方法 - 复用现有融合基础设施

  /// 检查两个循环是否具有相同的迭代结构
  static bool haveSameIterationStructure(AffineForOp loop1, AffineForOp loop2) {
    SmallVector<AffineForOp, 4> loops1, loops2;
    
    // 使用第一个内部操作来获取嵌套结构
    Operation *innerOp1 = nullptr;
    Operation *innerOp2 = nullptr;
    
    // 找到最内层的操作（非AffineForOp）
    loop1.walk([&](Operation *op) {
      if (!isa<AffineForOp, AffineYieldOp>(op) && !innerOp1) {
        innerOp1 = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    loop2.walk([&](Operation *op) {
      if (!isa<AffineForOp, AffineYieldOp>(op) && !innerOp2) {
        innerOp2 = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!innerOp1 || !innerOp2) {
      LLVM_DEBUG(llvm::dbgs() << "Could not find inner operations in loops\n");
      return false;
    }
    
    // 使用getAffineForIVs获取正确的嵌套结构
    getAffineForIVs(*innerOp1, &loops1);
    getAffineForIVs(*innerOp2, &loops2);
    
    LLVM_DEBUG(llvm::dbgs() << "Loop1 nesting depth: " << loops1.size() 
                            << ", Loop2 nesting depth: " << loops2.size() << "\n");
    
    if (loops1.size() != loops2.size()) {
      LLVM_DEBUG(llvm::dbgs() << "Different nesting depths\n");
      return false;
    }
    
    // 比较每一层的循环边界和步长
    for (size_t i = 0; i < loops1.size(); ++i) {
      AffineForOp forOp1 = loops1[i];
      AffineForOp forOp2 = loops2[i];
      
      if (forOp1.getLowerBoundMap() != forOp2.getLowerBoundMap() ||
          forOp1.getUpperBoundMap() != forOp2.getUpperBoundMap() ||
          forOp1.getStep() != forOp2.getStep()) {
        LLVM_DEBUG(llvm::dbgs() << "Loop bounds/step differ at depth " << i << "\n");
        return false;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Loop structures match\n");
    return true;
  }


  /// 检查两个循环是否有内存访问冲突
  static bool hasMemoryAccessConflict(unsigned srcId, unsigned dstId, MemRefDependenceGraph *mdg) {
    auto *srcNode = mdg->getNode(srcId);
    auto *dstNode = mdg->getNode(dstId);
    
    // 获取各自访问的memref
    DenseSet<Value> srcMemrefs, dstMemrefs;
    srcNode->getLoadAndStoreMemrefSet(&srcMemrefs);
    dstNode->getLoadAndStoreMemrefSet(&dstMemrefs);
    
    // 检查共享memref的访问模式
    for (Value srcMemref : srcMemrefs) {
      if (dstMemrefs.contains(srcMemref)) {
        // 如果任一循环对共享memref进行写操作，则存在冲突
        if (srcNode->getStoreOpCount(srcMemref) > 0 || 
            dstNode->getStoreOpCount(srcMemref) > 0) {

        // 尝试检查是否是不重叠的batch维度访问
        if (!hasBatchDimensionConflict(srcNode, dstNode, srcMemref)) {
          continue; // 不同batch，无冲突
        }

          return true;
        }
      }
    }
    
    return false;
  }

  // 辅助函数：检查两个节点对同一memref的batch维度是否有冲突
  static bool hasBatchDimensionConflict(MemRefDependenceGraph::Node *srcNode, 
                                        MemRefDependenceGraph::Node *dstNode,
                                        Value memref) {
    // 收集两个节点中对该memref的所有store操作
    SmallVector<Operation*> srcStores, dstStores;
    
    for (Operation *op : srcNode->stores) {
      if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        if (storeOp.getMemRef() == memref) {
          srcStores.push_back(op);
        }
      }
    }
    
    for (Operation *op : dstNode->stores) {
      if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        if (storeOp.getMemRef() == memref) {
          dstStores.push_back(op);
        }
      }
    }
    
    if (srcStores.empty() || dstStores.empty()) {
      return true; // 保守处理
    }
    // 分析第一个维度的访问范围
    for (Operation *srcOp : srcStores) {
      auto srcStore = cast<AffineStoreOp>(srcOp);
      auto srcIndices = srcStore.getMapOperands();
      
      for (Operation *dstOp : dstStores) {
        auto dstStore = cast<AffineStoreOp>(dstOp);
        auto dstIndices = dstStore.getMapOperands();
        
        // 检查第一个维度的访问模式
        if (!srcIndices.empty() && !dstIndices.empty()) {
          Value srcFirstDim = srcIndices[0];
          Value dstFirstDim = dstIndices[0];
          
          // 尝试判断两个访问的第一个维度是否不重叠
          if (areFirstDimensionDisjoint(srcStore, dstStore, srcFirstDim, dstFirstDim)) {
            continue; // 这对store不冲突，检查下一对
          }
        }
        
        // 如果无法证明不重叠，则认为有冲突
        return true;
      }
    }
    
    return false; // 所有store对都不冲突
  }

  // 判断两个store操作的第一个维度访问是否不重叠
  static bool areFirstDimensionDisjoint(AffineStoreOp srcStore, AffineStoreOp dstStore,
                                        Value srcFirstDim, Value dstFirstDim) {
    // 情况1: 直接使用不同的常量或参数
    if (srcFirstDim != dstFirstDim) {
      // 检查是否一个是直接使用循环变量，另一个是affine.apply的结果
      auto srcDefOp = srcFirstDim.getDefiningOp();
      auto dstDefOp = dstFirstDim.getDefiningOp();
      
      // 如果一个是affine.apply，另一个是直接的循环变量
      auto srcApply = dyn_cast_or_null<AffineApplyOp>(srcDefOp);
      auto dstApply = dyn_cast_or_null<AffineApplyOp>(dstDefOp);
      
      // 情况1a: 一个是循环变量，一个是affine.apply
      if ((srcApply && !dstApply) || (!srcApply && dstApply)) {
        return true; // 简单情况：很可能是不同的batch段
      }
      
      // 情况1b: 两个都是affine.apply但map不同
      if (srcApply && dstApply) {
        if (srcApply.getAffineMap() != dstApply.getAffineMap()) {
          // 检查它们的输入是否相同
          auto srcOperands = srcApply.getMapOperands();
          auto dstOperands = dstApply.getMapOperands();
          
          // 如果使用相同的循环变量但不同的map，通常意味着不同的batch段
          if (!srcOperands.empty() && !dstOperands.empty() && 
              srcOperands[0] == dstOperands[0]) {
            return true;
          }
        }
      }
    }
    
    // 情况2: 使用依赖分析进行更精确的判断（可选）
    // 可以使用 MLIR 的依赖分析工具来检查访问范围是否重叠
    // 这里暂时采用保守策略
    
    return false; // 无法证明不重叠
  }

  /// 获取独立循环的融合候选 - 类似getProducerCandidates但针对独立循环
  static void getIndependentCandidates(unsigned dstId, MemRefDependenceGraph *mdg,
                                      SmallVectorImpl<unsigned> &srcIdCandidates) {
    auto *dstNode = mdg->getNode(dstId);
    auto dstLoop = cast<AffineForOp>(dstNode->op);
    // 遍历所有其他节点寻找独立的循环
    for (auto &srcNodePair : mdg->nodes) {
      unsigned srcId = srcNodePair.first;
      auto *srcNode = &srcNodePair.second;
      
      // 跳过自己和非循环节点
      if (srcId == dstId || !isa<AffineForOp>(srcNode->op))
        continue;
      
      auto srcLoop = cast<AffineForOp>(srcNode->op);
      
      // 检查是否在同一个block中
      if (srcLoop->getBlock() != dstLoop->getBlock())
        continue;
      
      // 检查循环结构是否相同（你之前提到的关键条件）
      if (!haveSameIterationStructure(srcLoop, dstLoop))
        continue;
      // 检查是否真正独立（无依赖关系）
      if (mdg->hasDependencePath(srcId, dstId) || mdg->hasDependencePath(dstId, srcId))
        continue;
      // 检查内存访问是否冲突
      if (hasMemoryAccessConflict(srcId, dstId, mdg))
        continue;
      
      srcIdCandidates.push_back(srcId);
    }
    
    llvm::sort(srcIdCandidates);
    srcIdCandidates.erase(llvm::unique(srcIdCandidates), srcIdCandidates.end());
  }

  // /// 检查循环是否适合独立融合（element-wise操作等）
  // bool isEligibleForIndependentFusion(AffineForOp loop) {
  //   // 检查是否包含简单的element-wise操作
  //   bool isElementWise = true;
  //   loop.walk([&](Operation *op) {
  //     if (isa<AffineForOp, AffineYieldOp, AffineLoadOp, AffineStoreOp>(op))
  //       return WalkResult::advance();
      
  //     if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
  //             math::SqrtOp, math::ExpOp, math::TanhOp>(op))
  //       return WalkResult::advance();
      
  //     if (op->hasTrait<OpTrait::ConstantLike>())
  //       return WalkResult::advance();
      
  //     // 其他复杂操作不适合
  //     isElementWise = false;
  //     return WalkResult::interrupt();
  //   });
    
  //   return isElementWise;
  // }

  // 修复1: 添加相邻性检查函数
  static bool areLoopsAdjacent(AffineForOp loop1, AffineForOp loop2) {
    Operation *op1 = loop1.getOperation();
    Operation *op2 = loop2.getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "Checking adjacency between two loops\n");
    
    // 检查是否在同一个block中
    if (op1->getBlock() != op2->getBlock()) {
      LLVM_DEBUG(llvm::dbgs() << "Loops are not in the same block\n");
      return false;
    }
    
    // 检查op1是否在op2之前
    if (!op1->isBeforeInBlock(op2)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop1 is not before Loop2\n");
      return false;
    }
    
    // 检查是否直接相邻（中间只能有alloc等非循环操作）
    Operation *next = op1->getNextNode();
    int operationsBetween = 0;
    
    while (next && next != op2) {
      operationsBetween++;
      LLVM_DEBUG(llvm::dbgs() << "Operation between loops: " << *next << "\n");
      
      // 如果中间有其他AffineForOp，则不相邻
      if (isa<AffineForOp>(next)) {
        LLVM_DEBUG(llvm::dbgs() << "Found another AffineForOp between the loops\n");
        return false;
      }
      
      next = next->getNextNode();
    }
    
    bool adjacent = (next == op2);
    LLVM_DEBUG(llvm::dbgs() << "Operations between loops: " << operationsBetween 
                            << ", Adjacent: " << adjacent << "\n");
    
    return adjacent;
  }

  // // 修复2: 简化的相邻候选查找函数
  // static void getAdjacentIndependentCandidates(unsigned dstId, MemRefDependenceGraph *mdg,
  //                                           SmallVectorImpl<unsigned> &srcIdCandidates) {
  //   auto *dstNode = mdg->getNode(dstId);
  //   if (!dstNode) {
  //     LLVM_DEBUG(llvm::dbgs() << "Destination node not found: " << dstId << "\n");
  //     return;
  //   }
    
  //   auto dstLoop = dyn_cast<AffineForOp>(dstNode->op);
  //   if (!dstLoop) {
  //     LLVM_DEBUG(llvm::dbgs() << "Destination is not an AffineForOp\n");
  //     return;
  //   }
    
  //   // 只检查前一个操作，寻找相邻的循环
  //   Operation *prevOp = dstLoop->getPrevNode();
  //   while (prevOp) {
  //     // 如果遇到另一个affine.for循环
  //     if (auto srcLoop = dyn_cast<AffineForOp>(prevOp)) {
  //       auto *srcNode = mdg->getForOpNode(srcLoop);
  //       if (!srcNode) {
  //         LLVM_DEBUG(llvm::dbgs() << "Source node not found in MDG\n");
  //         break;
  //       }
        
  //       unsigned srcId = srcNode->id;
        
  //       // 检查是否真正相邻
  //       if (!areLoopsAdjacent(srcLoop, dstLoop)) {
  //         LLVM_DEBUG(llvm::dbgs() << "Loops are not adjacent\n");
  //         break;
  //       }
        
  //       // 检查循环结构是否相同
  //       if (!haveSameIterationStructure(srcLoop, dstLoop)) {
  //         LLVM_DEBUG(llvm::dbgs() << "Loop structures differ\n");
  //         break;
  //       }
        
  //       // 检查是否真正独立（无依赖关系）
  //       if (mdg->hasDependencePath(srcId, dstId) || mdg->hasDependencePath(dstId, srcId)) {
  //         LLVM_DEBUG(llvm::dbgs() << "Loops have dependence path\n");
  //         break;
  //       }
        
  //       // 简化的内存访问检查：只检查写-写冲突
  //       if (hasWriteWriteConflict(srcId, dstId, mdg)) {
  //         LLVM_DEBUG(llvm::dbgs() << "Loops have write-write conflict\n");
  //         break;
  //       }
        
  //       srcIdCandidates.push_back(srcId);
  //       LLVM_DEBUG(llvm::dbgs() << "Found adjacent candidate: " << srcId << "\n");
  //       break;  // 只考虑直接相邻的一个循环
  //     }
      
  //     // 如果遇到其他affine.for循环，停止搜索
  //     if (isa<AffineForOp>(prevOp))
  //       break;
        
  //     prevOp = prevOp->getPrevNode();
  //   }
  // }

  // 修复3: 简化的写-写冲突检查
  static bool hasWriteWriteConflict(unsigned srcId, unsigned dstId, MemRefDependenceGraph *mdg) {
    auto *srcNode = mdg->getNode(srcId);
    auto *dstNode = mdg->getNode(dstId);
    
    if (!srcNode || !dstNode)
      return true;  // 保守策略：如果节点不存在，认为有冲突
    
    // 获取写入的memref
    DenseSet<Value> srcWrites, dstWrites;
    for (Operation *storeOp : srcNode->stores) {
      auto writeOp = dyn_cast<AffineWriteOpInterface>(storeOp);
      if (writeOp)
        srcWrites.insert(writeOp.getMemRef());
    }
    for (Operation *storeOp : dstNode->stores) {
      auto writeOp = dyn_cast<AffineWriteOpInterface>(storeOp);
      if (writeOp)
        dstWrites.insert(writeOp.getMemRef());
    }
    
    // 检查是否有写-写冲突
    for (Value srcMemref : srcWrites) {
      // 尝试检查是否是不重叠的batch维度访问
      if (!hasBatchDimensionConflict(srcNode, dstNode, srcMemref)) {
        continue; // 不同batch，无冲突
      }

      if (dstWrites.contains(srcMemref)) {
        return true;
      }
    }
    
    return false;
  }

  bool fuseIndependentLoops(AffineForOp srcLoop, AffineForOp dstLoop) {
    LLVM_DEBUG(llvm::dbgs() << "Attempting to fuse independent loops\n");
    
    // 找到最内层的非循环操作来获取完整的嵌套结构
    Operation *srcInnerOp = nullptr;
    Operation *dstInnerOp = nullptr;
    
    srcLoop.walk([&](Operation *op) {
      if (!isa<AffineForOp, AffineYieldOp>(op) && !srcInnerOp) {
        srcInnerOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    dstLoop.walk([&](Operation *op) {
      if (!isa<AffineForOp, AffineYieldOp>(op) && !dstInnerOp) {
        dstInnerOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!srcInnerOp || !dstInnerOp) {
      LLVM_DEBUG(llvm::dbgs() << "Could not find inner operations\n");
      return false;
    }
    
    // 使用getAffineForIVs获取正确的循环嵌套
    SmallVector<AffineForOp, 4> srcLoops, dstLoops;
    getAffineForIVs(*srcInnerOp, &srcLoops);
    getAffineForIVs(*dstInnerOp, &dstLoops);
    
    if (srcLoops.size() != dstLoops.size()) {
      LLVM_DEBUG(llvm::dbgs() << "Nesting depths differ: src=" << srcLoops.size() 
                              << ", dst=" << dstLoops.size() << "\n");
      return false;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Collected " << srcLoops.size() << " nested loops\n");
    
    // 创建IRMapping - 只映射归纳变量，让外部值保持原样
    IRMapping valueMapping;
    
    // 映射所有层次的归纳变量
    for (size_t i = 0; i < srcLoops.size(); ++i) {
      Value srcIV = srcLoops[i].getInductionVar();
      Value dstIV = dstLoops[i].getInductionVar();
      valueMapping.map(srcIV, dstIV);
      LLVM_DEBUG(llvm::dbgs() << "Mapping IV " << i << ": " << srcIV << " -> " << dstIV << "\n");
    }
    
    // 获取最内层循环进行body合并
    AffineForOp srcInnerLoop = srcLoops.back();
    AffineForOp dstInnerLoop = dstLoops.back();
    
    // 创建builder
    OpBuilder builder(dstInnerLoop);
    builder.setInsertionPoint(dstInnerLoop.getBody()->getTerminator());
    
    // 克隆源循环最内层body中的所有操作
    for (Operation &op : srcInnerLoop.getBody()->without_terminator()) {
      Operation *clonedOp = builder.clone(op, valueMapping);
      LLVM_DEBUG(llvm::dbgs() << "Cloned operation: " << op << "\n");
      (void)clonedOp; // 避免未使用变量警告
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully fused independent loops\n");
    return true;
  }

  /// 独立循环融合的主函数 - 复用现有融合逻辑
  // void runIndependentFusion() {
  //   LLVM_DEBUG(llvm::dbgs() << "--- Independent Loop Fusion ---\n");
  //   init();
    
  //   while (!worklist.empty()) {
  //     unsigned dstId = worklist.back();
  //     worklist.pop_back();
      
  //     // 跳过已删除的节点
  //     if (mdg->nodes.count(dstId) == 0)
  //       continue;
      
  //     auto *dstNode = mdg->getNode(dstId);
  //     if (!isa<AffineForOp>(dstNode->op))
  //       continue;
      
  //     auto dstAffineForOp = cast<AffineForOp>(dstNode->op);
      
  //     // // 检查目标循环是否适合融合
  //     // if (!isEligibleForIndependentFusion(dstAffineForOp))
  //     //   continue;
      
  //     LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << " for independent fusion\n");
      
  //     // 获取独立循环候选
  //     SmallVector<unsigned, 16> srcIdCandidates;
  //     getIndependentCandidates(dstId, mdg, srcIdCandidates);
      
  //     for (unsigned srcId : srcIdCandidates) {
  //       auto *srcNode = mdg->getNode(srcId);
  //       if (!srcNode) continue;
        
  //       auto srcAffineForOp = cast<AffineForOp>(srcNode->op);
        
  //       // // 检查源循环是否适合融合
  //       // if (!isEligibleForIndependentFusion(srcAffineForOp))
  //       //   continue;
        
  //       LLVM_DEBUG(llvm::dbgs() << "Attempting to fuse src loop " << srcId 
  //                               << " into dst loop " << dstId << "\n");
        
  //       // 计算融合点
  //       Operation *fusedLoopInsPoint =
  //           mdg->getFusedLoopNestInsertionPoint(srcNode->id, dstNode->id);
  //       if (fusedLoopInsPoint == nullptr)
  //         continue;
        
  //       // 获取公共循环深度信息
  //       SmallVector<AffineForOp, 4> surroundingLoops;
  //       getAffineForIVs(*dstAffineForOp, &surroundingLoops);
  //       unsigned numSurroundingLoops = surroundingLoops.size();
        
  //       // 尝试在不同深度进行融合
  //       unsigned maxLegalFusionDepth = 0;
  //       SmallVector<ComputationSliceState, 8> depthSliceUnions;
  //       depthSliceUnions.resize(surroundingLoops.size());
        
  //       // 使用Generic策略进行可行性检查
  //       FusionStrategy strategy(FusionStrategy::Generic);
        
  //       for (unsigned i = 1; i <= surroundingLoops.size(); ++i) {
  //         FusionResult result =
  //             affine::canFuseLoops(srcAffineForOp, dstAffineForOp,
  //                               /*dstLoopDepth=*/i + numSurroundingLoops,
  //                               &depthSliceUnions[i - 1], strategy);

  //         if (result.value == FusionResult::Success)
  //           maxLegalFusionDepth = i;
  //       }

  //       if (maxLegalFusionDepth == 0) {
  //         LLVM_DEBUG(llvm::dbgs() << "Fusion not legal at any depth\n");
  //         continue;
  //       }
        
  //       // 选择最深的合法融合深度
  //       unsigned bestDstLoopDepth = maxLegalFusionDepth;
  //       ComputationSliceState &bestSlice = depthSliceUnions[bestDstLoopDepth - 1];
        
  //       // 使用MLIR内置的融合函数执行融合
  //       fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Successfully fused src loop " << srcId 
  //                               << " into dst loop " << dstId << " at depth " << bestDstLoopDepth << "\n");
        
  //       // 移动融合后的循环到正确位置
  //       if (fusedLoopInsPoint != dstAffineForOp)
  //         dstAffineForOp->moveBefore(fusedLoopInsPoint);
        
  //       // 更新依赖图 - 复用现有逻辑
  //       mdg->updateEdges(srcNode->id, dstNode->id, /*privateMemrefs=*/{}, /*removeSrcNode=*/true);
        
  //       // 重新收集目标循环状态
  //       LoopNestStateCollector dstLoopCollector;
  //       dstLoopCollector.collect(dstAffineForOp);
        
  //       mdg->clearNodeLoadAndStores(dstId);
  //       mdg->addToNode(dstId, dstLoopCollector.loadOpInsts, dstLoopCollector.storeOpInsts);
        
  //       // 删除源循环
  //       srcAffineForOp.erase();
  //       mdg->removeNode(srcId);
        
  //       // 成功融合一个后，重新开始（因为图结构已改变）
  //       break;
  //     }
  //   }
  // }

  // //这个版本在处理整段IR的时候，存在删除问题
  // void runIndependentFusion() {
  //   LLVM_DEBUG(llvm::dbgs() << "--- Independent Loop Fusion ---\n");
  //   init();
    
  //   while (!worklist.empty()) {
  //     unsigned dstId = worklist.back();
  //     worklist.pop_back();
      
  //     // 跳过已删除的节点
  //     if (mdg->nodes.count(dstId) == 0) {
  //       LLVM_DEBUG(llvm::dbgs() << "Skipping removed node: " << dstId << "\n");
  //       continue;
  //     }
      
  //     auto *dstNode = mdg->getNode(dstId);
  //     if (!isa<AffineForOp>(dstNode->op)) {
  //       LLVM_DEBUG(llvm::dbgs() << "Skipping non-AffineForOp node: " << dstId << "\n");
  //       continue;
  //     }
      
  //     auto dstAffineForOp = cast<AffineForOp>(dstNode->op);
      
  //     LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << " for independent fusion\n");
      
  //     // 获取相邻的独立循环候选
  //     SmallVector<unsigned, 4> srcIdCandidates;
  //     getAdjacentIndependentCandidates(dstId, mdg, srcIdCandidates);
      
  //     if (srcIdCandidates.empty()) {
  //       LLVM_DEBUG(llvm::dbgs() << "No adjacent candidates found for " << dstId << "\n");
  //       continue;
  //     }
      
  //     for (unsigned srcId : srcIdCandidates) {
  //       auto *srcNode = mdg->getNode(srcId);
  //       if (!srcNode) {
  //         LLVM_DEBUG(llvm::dbgs() << "Source node not found: " << srcId << "\n");
  //         continue;
  //       }
        
  //       auto srcAffineForOp = cast<AffineForOp>(srcNode->op);
        
  //       LLVM_DEBUG(llvm::dbgs() << "Attempting to fuse adjacent src loop " << srcId 
  //                               << " into dst loop " << dstId << "\n");
        
  //       // 对于独立循环，直接进行简单合并
  //       if (fuseIndependentLoops(srcAffineForOp, dstAffineForOp)) {
  //         LLVM_DEBUG(llvm::dbgs() << "Successfully fused adjacent loops " << srcId 
  //                                 << " and " << dstId << "\n");
          
  //         // 更新依赖图
  //         mdg->updateEdges(srcNode->id, dstNode->id, /*privateMemrefs=*/{}, /*removeSrcNode=*/true);
          
  //         // 重新收集目标循环状态
  //         LoopNestStateCollector dstLoopCollector;
  //         dstLoopCollector.collect(dstAffineForOp);
          
  //         mdg->clearNodeLoadAndStores(dstId);
  //         mdg->addToNode(dstId, dstLoopCollector.loadOpInsts, dstLoopCollector.storeOpInsts);
          
  //         // 删除源循环
  //         srcAffineForOp.erase();
  //         mdg->removeNode(srcId);
          
  //         // 成功融合后，重新开始
  //         break;
  //       } else {
  //         LLVM_DEBUG(llvm::dbgs() << "Failed to fuse loops " << srcId << " and " << dstId << "\n");
  //       }
  //     }
  //   }
    
  //   LLVM_DEBUG(llvm::dbgs() << "Independent fusion completed\n");
  // }

  // 添加辅助函数：检查Value是否仍有使用者
  static bool hasActiveUsers(Value value) {
    return !value.use_empty();
  }

  // 添加辅助函数：安全删除操作
  static void safeEraseOperation(Operation *op) {
    LLVM_DEBUG(llvm::dbgs() << "Attempting to safely erase operation: " << *op << "\n");
    
    // 检查操作的所有结果是否还有使用者
    for (Value result : op->getResults()) {
      if (hasActiveUsers(result)) {
        LLVM_DEBUG(llvm::dbgs() << "Warning: Operation still has active users: " << *op << "\n");
        // 可以选择不删除，或者进一步处理
        return;
      }
    }
    
    // 递归检查嵌套操作
    if (op->getNumRegions() > 0) {
      op->walk([&](Operation *nestedOp) {
        if (nestedOp != op) {
          for (Value result : nestedOp->getResults()) {
            if (hasActiveUsers(result)) {
              LLVM_DEBUG(llvm::dbgs() << "Warning: Nested operation still has active users: " << *nestedOp << "\n");
            }
          }
        }
        return WalkResult::advance();
      });
    }
    
    // 如果所有检查都通过，安全删除
    op->erase();
    LLVM_DEBUG(llvm::dbgs() << "Successfully erased operation\n");
  }

  // 检查操作是否允许存在于相邻循环之间
  static bool isAllowedBetweenAdjacentLoops(Operation *op) {
    // 1. 内存分配操作
    if (isa<memref::AllocOp>(op) || isa<memref::AllocaOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Allowed: memory allocation\n");
      return true;
    }
    
    // 2. 常量操作
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Allowed: constant operation\n");
      return true;
    }
    
    // 3. arith常量操作
    if (isa<arith::ConstantOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Allowed: arith constant\n");
      return true;
    }
    
    // 4. memref.reinterpret_cast 操作
    if (isa<memref::ReinterpretCastOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "  -> Allowed: memref reinterpret_cast\n");
      return true;
    }
    
    // 不允许的操作
    LLVM_DEBUG(llvm::dbgs() << "  -> Disallowed: " << op->getName() << "\n");
    return false;
  }

  // 允许中间有alloc和constant的相邻性检查
  static bool areLoopsAdjacentWithAlloc(AffineForOp loop1, AffineForOp loop2) {
    Operation *op1 = loop1.getOperation();
    Operation *op2 = loop2.getOperation();
    
    LLVM_DEBUG(llvm::dbgs() << "Checking adjacency (with alloc tolerance) between two loops\n");
    
    // 检查是否在同一个block中
    if (op1->getBlock() != op2->getBlock()) {
      LLVM_DEBUG(llvm::dbgs() << "Loops are not in the same block\n");
      return false;
    }
    
    // 检查op1是否在op2之前
    if (!op1->isBeforeInBlock(op2)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop1 is not before Loop2\n");
      return false;
    }
    
    // 检查中间的操作是否都是允许的
    Operation *next = op1->getNextNode();
    int operationsBetween = 0;
    int allowedOps = 0;
    int disallowedOps = 0;
    
    while (next && next != op2) {
      operationsBetween++;
      LLVM_DEBUG(llvm::dbgs() << "Operation between loops: " << *next << "\n");
      
      if (isAllowedBetweenAdjacentLoops(next)) {
        allowedOps++;
      } else {
        disallowedOps++;
        LLVM_DEBUG(llvm::dbgs() << "Found disallowed operation: " << next->getName() << "\n");
        return false;
      }
      
      next = next->getNextNode();
    }
    
    bool adjacent = (next == op2);
    LLVM_DEBUG(llvm::dbgs() << "Operations between loops: " << operationsBetween 
                            << " (allowed: " << allowedOps << ", disallowed: " << disallowedOps << ")"
                            << ", Adjacent: " << adjacent << "\n");
    
    return adjacent;
  }

  // 相邻候选查找函数 - 允许中间有alloc和constant
  static void getAdjacentIndependentCandidates(unsigned dstId, MemRefDependenceGraph *mdg,
                                            SmallVectorImpl<unsigned> &srcIdCandidates) {                                        
    auto *dstNode = mdg->getNode(dstId);
    if (!dstNode) {
      LLVM_DEBUG(llvm::dbgs() << "Destination node not found: " << dstId << "\n");
      return;
    }
    
    auto dstLoop = dyn_cast<AffineForOp>(dstNode->op);
    if (!dstLoop) {
      LLVM_DEBUG(llvm::dbgs() << "Destination is not an AffineForOp\n");
      return;
    }
    
    // 向前搜索AffineForOp，跳过允许的中间操作
    Operation *currentOp = dstLoop->getPrevNode();
    
    // 跳过允许的操作直到找到AffineForOp或遇到不允许的操作
    while (currentOp) {
      LLVM_DEBUG(llvm::dbgs() << "Examining previous operation: " << *currentOp << "\n");
      
      if (auto srcLoop = dyn_cast<AffineForOp>(currentOp)) {
        // 找到了一个AffineForOp，检查是否可以融合
        auto *srcNode = mdg->getForOpNode(srcLoop);
        if (!srcNode) {
          LLVM_DEBUG(llvm::dbgs() << "Source node not found in MDG\n");
          break;
        }
        
        unsigned srcId = srcNode->id;
        
        // 使用允许alloc的相邻性检查
        if (!areLoopsAdjacentWithAlloc(srcLoop, dstLoop)) {
          LLVM_DEBUG(llvm::dbgs() << "Loops are not adjacent (even with alloc tolerance)\n");
          break;
        }
        
        // 检查循环结构是否相同
        if (!haveSameIterationStructure(srcLoop, dstLoop)) {
          LLVM_DEBUG(llvm::dbgs() << "Loop structures differ\n");
          break;
        }
        
        // 检查是否真正独立（无依赖关系）
        if (mdg->hasDependencePath(srcId, dstId) || mdg->hasDependencePath(dstId, srcId)) {
          LLVM_DEBUG(llvm::dbgs() << "Loops have dependence path\n");
          break;
        }
        
        // 检查写-写冲突
        if (hasWriteWriteConflict(srcId, dstId, mdg)) {
          LLVM_DEBUG(llvm::dbgs() << "Loops have write-write conflict\n");
          break;
        }
        
        srcIdCandidates.push_back(srcId);
        LLVM_DEBUG(llvm::dbgs() << "Found adjacent candidate (with alloc tolerance): " << srcId << "\n");
        break; // 只考虑最近的一个候选循环
        
      } else if (isAllowedBetweenAdjacentLoops(currentOp)) {
        // 这是允许的中间操作，继续向前搜索
        LLVM_DEBUG(llvm::dbgs() << "Skipping allowed intermediate operation\n");
        currentOp = currentOp->getPrevNode();
      } else {
        // 遇到了不允许的操作，停止搜索
        LLVM_DEBUG(llvm::dbgs() << "Encountered disallowed operation, stopping search\n");
        break;
      }
    }
  }

  static void safeDeleteLoop(AffineForOp loopToDelete, MemRefDependenceGraph *mdg, unsigned nodeId) {
    LLVM_DEBUG(llvm::dbgs() << "Safely deleting loop with node ID: " << nodeId << "\n");
    
    // 1. 首先检查是否还有未处理的Value使用
    bool hasUnresolvedUses = false;
    loopToDelete.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          LLVM_DEBUG(llvm::dbgs() << "Found unresolved use of value: " << result << "\n");
          for (Operation *user : result.getUsers()) {
            LLVM_DEBUG(llvm::dbgs() << "  User: " << *user << "\n");
          }
          hasUnresolvedUses = true;
        }
      }
      return WalkResult::advance();
    });
    
    if (hasUnresolvedUses) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot delete loop: has unresolved uses\n");
      return;
    }
    
    // 2. 从依赖图中清理
    mdg->clearNodeLoadAndStores(nodeId);
    mdg->removeNode(nodeId);
    
    // 3. 删除循环操作
    loopToDelete.erase();
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully deleted loop\n");
  }

  //修复相邻性的识别问题
  void runIndependentFusion() {
    LLVM_DEBUG(llvm::dbgs() << "--- Independent Loop Fusion (Simplified) ---\n");
    init();
    
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      
      if (mdg->nodes.count(dstId) == 0) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping removed node: " << dstId << "\n");
        continue;
      }
      
      auto *dstNode = mdg->getNode(dstId);
      if (!isa<AffineForOp>(dstNode->op)) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping non-AffineForOp node: " << dstId << "\n");
        continue;
      }
      
      auto dstAffineForOp = cast<AffineForOp>(dstNode->op);
      
      LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << " for independent fusion\n");
      
      SmallVector<unsigned, 4> srcIdCandidates;
      getAdjacentIndependentCandidates(dstId, mdg, srcIdCandidates);
      
      if (srcIdCandidates.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "No adjacent candidates found for " << dstId << "\n");
        continue;
      }
      
      for (unsigned srcId : srcIdCandidates) {
        auto *srcNode = mdg->getNode(srcId);
        if (!srcNode) {
          LLVM_DEBUG(llvm::dbgs() << "Source node not found: " << srcId << "\n");
          continue;
        }
        
        auto srcAffineForOp = cast<AffineForOp>(srcNode->op);
        
        LLVM_DEBUG(llvm::dbgs() << "Attempting to fuse adjacent src loop " << srcId 
                                << " into dst loop " << dstId << "\n");
        
        // 执行融合
        if (fuseIndependentLoops(srcAffineForOp, dstAffineForOp)) {
          LLVM_DEBUG(llvm::dbgs() << "Successfully fused adjacent loops " << srcId 
                                  << " and " << dstId << "\n");
          
          // 更新目标循环的状态
          LoopNestStateCollector dstLoopCollector;
          dstLoopCollector.collect(dstAffineForOp);
          
          mdg->clearNodeLoadAndStores(dstId);
          mdg->addToNode(dstId, dstLoopCollector.loadOpInsts, dstLoopCollector.storeOpInsts);
          
          // 更新依赖图
          mdg->updateEdges(srcNode->id, dstNode->id, /*privateMemrefs=*/{}, /*removeSrcNode=*/false);
          
          // 清理源节点并删除源循环
          mdg->clearNodeLoadAndStores(srcId);
          mdg->removeNode(srcId);
          srcAffineForOp.erase();
          
          LLVM_DEBUG(llvm::dbgs() << "Completed fusion and cleanup\n");
          break;
        } else {
          LLVM_DEBUG(llvm::dbgs() << "Failed to fuse loops " << srcId << " and " << dstId << "\n");
        }
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Independent fusion completed\n");
  }

  /// 检查两个affine.for循环嵌套是否具有完全相同的结构
  /// 包括嵌套深度、每层循环的边界条件等 add py p
  static bool haveSameIterationCounts(AffineForOp srcLoop, AffineForOp dstLoop) {
    SmallVector<AffineForOp, 4> srcLoops, dstLoops;
    getAffineForIVs(*srcLoop.getOperation(), &srcLoops);
    getAffineForIVs(*dstLoop.getOperation(), &dstLoops);
    
    // 检查嵌套深度
    if (srcLoops.size() != dstLoops.size()) {
      LLVM_DEBUG(llvm::dbgs() << "Loop nest depths differ: src=" << srcLoops.size() 
                              << ", dst=" << dstLoops.size() << "\n");
      return false;
    }
    
    // 检查每层的迭代次数
    for (size_t i = 0; i < srcLoops.size(); ++i) {
      AffineForOp srcForOp = srcLoops[i];
      AffineForOp dstForOp = dstLoops[i];
      
      // 方案1：先检查是否有常量边界，然后获取值
      if (!srcForOp.hasConstantLowerBound() || !srcForOp.hasConstantUpperBound() ||
          !dstForOp.hasConstantLowerBound() || !dstForOp.hasConstantUpperBound()) {
        LLVM_DEBUG(llvm::dbgs() << "Non-constant bounds at depth " << i << ", skipping check\n");
        continue;
      }
      
      // 直接获取常量边界值
      int64_t srcLower = srcForOp.getConstantLowerBound();
      int64_t srcUpper = srcForOp.getConstantUpperBound();
      int64_t dstLower = dstForOp.getConstantLowerBound();
      int64_t dstUpper = dstForOp.getConstantUpperBound();
      
      // 检查步长
      llvm::APInt srcStep = srcForOp.getStep();
      llvm::APInt dstStep = dstForOp.getStep();
      
      if (srcStep != dstStep) {
        LLVM_DEBUG(llvm::dbgs() << "Steps differ at depth " << i << "\n");
        return false;
      }
      
      // 计算迭代次数
      int64_t srcIterCount = srcUpper - srcLower;
      int64_t dstIterCount = dstUpper - dstLower;
      
      if (srcIterCount != dstIterCount) {
        LLVM_DEBUG(llvm::dbgs() << "Iteration counts differ at depth " << i 
                                << ": src=" << srcIterCount << ", dst=" << dstIterCount << "\n");
        return false;
      }
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Loop iteration counts match\n");
    return true;
  }

  /// Perform fusions with node `dstId` as the destination of fusion, with
  /// No fusion is performed when producers with a user count greater than
  /// `maxSrcUserCount` for any of the memrefs involved.
  void performFusionsIntoDest(unsigned dstId, unsigned maxSrcUserCount) {
    LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");
    // Skip if this node was removed (fused into another node).
    if (mdg->nodes.count(dstId) == 0)
      return;
    // Get 'dstNode' into which to attempt fusion.
    auto *dstNode = mdg->getNode(dstId);
    // Skip if 'dstNode' is not a loop nest.
    if (!isa<AffineForOp>(dstNode->op))
      return;
    // Skip if 'dstNode' is a loop nest returning values.
    // TODO: support loop nests that return values.
    if (dstNode->op->getNumResults() > 0)
      return;

    LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");

    // Sink sequential loops in 'dstNode' (and thus raise parallel loops)
    // while preserving relative order. This can increase the maximum loop
    // depth at which we can fuse a slice of a producer loop nest into a
    // consumer loop nest.
    sinkSequentialLoops(dstNode);
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    // Try to fuse 'dstNode' with candidate producer loops until a fixed point
    // is reached. Fusing two loops may expose new fusion opportunities.
    bool dstNodeChanged;
    do {
      // Gather src loop candidates for 'dstNode' and visit them in "quasi"
      // reverse program order to minimize the number of iterations needed to
      // reach the fixed point. Note that this is a best effort approach since
      // 'getProducerCandidates' does not always guarantee that program order
      // in 'srcIdCandidates'.
      dstNodeChanged = false;
      SmallVector<unsigned, 16> srcIdCandidates;
      getProducerCandidates(dstId, mdg, srcIdCandidates);

      for (unsigned srcId : llvm::reverse(srcIdCandidates)) {
        // Get 'srcNode' from which to attempt fusion into 'dstNode'.
        auto *srcNode = mdg->getNode(srcId);
        auto srcAffineForOp = cast<AffineForOp>(srcNode->op);
        LLVM_DEBUG(llvm::dbgs() << "Evaluating src loop " << srcId
                                << " for dst loop " << dstId << "\n");

        // Skip if 'srcNode' is a loop nest returning values.
        // TODO: support loop nests that return values.
        if (isa<AffineForOp>(srcNode->op) && srcNode->op->getNumResults() > 0)
          continue;

        // // 检查两个循环嵌套是否具有相同的结构 add py p
        // if (!haveSameIterationCounts(srcAffineForOp, dstAffineForOp)) {
        //   LLVM_DEBUG(llvm::dbgs() << "Skipping fusion: loop structures differ between src " 
        //                           << srcId << " and dst " << dstId << "\n");
        //   continue;
        // }

        DenseSet<Value> producerConsumerMemrefs;
        gatherProducerConsumerMemrefs(srcId, dstId, mdg,
                                      producerConsumerMemrefs);

        // Skip if 'srcNode' out edge count on any memref is greater than
        // 'maxSrcUserCount'.
        if (any_of(producerConsumerMemrefs, [&](Value memref) {
              return mdg->getOutEdgeCount(srcNode->id, memref) >
                     maxSrcUserCount;
            }))
          continue;

        // Gather memrefs in 'srcNode' that are written and escape out of the
        // block (e.g., memref block arguments, returned memrefs,
        // memrefs passed to function calls, etc.).
        DenseSet<Value> srcEscapingMemRefs;
        // gatherEscapingMemrefs(srcNode->id, mdg, srcEscapingMemRefs); //modified py p

        //modified py p
        // auto *srcNode = mdg->getNode(srcId);
        auto *dstNode = mdg->getNode(dstId);
        
        for (Operation *storeOp : srcNode->stores) {
          auto memref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
          if (srcEscapingMemRefs.count(memref))
            continue;
          // 传入srcOp和dstOp进行精细分析
          if (isEscapingMemref(memref, &mdg->block, srcNode->op, dstNode->op))
            srcEscapingMemRefs.insert(memref);
        }

        // Skip if there are non-affine operations in between the 'srcNode'
        // and 'dstNode' using their memrefs. If so, we wouldn't be able to
        // compute a legal insertion point for now. 'srcNode' and 'dstNode'
        // memrefs with non-affine operation users would be considered
        // escaping memrefs so we can limit this check to only scenarios with
        // escaping memrefs.
        if (!srcEscapingMemRefs.empty() &&
            hasNonAffineUsersOnThePath(srcId, dstId, mdg)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Can't fuse: non-affine users in between the loops\n");
          continue;
        }

        // Compute an operation list insertion point for the fused loop
        // nest which preserves dependences.
        Operation *fusedLoopInsPoint =
            mdg->getFusedLoopNestInsertionPoint(srcNode->id, dstNode->id);
        if (fusedLoopInsPoint == nullptr)
          continue;

        // It's possible this fusion is at an inner depth (i.e., there are
        // common surrounding affine loops for the source and destination for
        // ops). We need to get this number because the call to canFuseLoops
        // needs to be passed the absolute depth. The max legal depth and the
        // depths we try below are however *relative* and as such don't include
        // the common depth.
        SmallVector<AffineForOp, 4> surroundingLoops;
        getAffineForIVs(*dstAffineForOp, &surroundingLoops);
        unsigned numSurroundingLoops = surroundingLoops.size();

        // Compute the innermost common loop depth for dstNode
        // producer-consumer loads/stores.
        SmallVector<Operation *, 2> dstMemrefOps;
        for (Operation *op : dstNode->loads)
          if (producerConsumerMemrefs.count(
                  cast<AffineReadOpInterface>(op).getMemRef()) > 0)
            dstMemrefOps.push_back(op);
        for (Operation *op : dstNode->stores)
          if (producerConsumerMemrefs.count(
                  cast<AffineWriteOpInterface>(op).getMemRef()))
            dstMemrefOps.push_back(op);
        unsigned dstLoopDepthTest =
            getInnermostCommonLoopDepth(dstMemrefOps) - numSurroundingLoops;

        // Check the feasibility of fusing src loop nest into dst loop nest
        // at loop depths in range [1, dstLoopDepthTest].
        unsigned maxLegalFusionDepth = 0;
        SmallVector<ComputationSliceState, 8> depthSliceUnions;
        depthSliceUnions.resize(dstLoopDepthTest);
        FusionStrategy strategy(FusionStrategy::ProducerConsumer);
        for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
          FusionResult result =
              affine::canFuseLoops(srcAffineForOp, dstAffineForOp,
                                   /*dstLoopDepth=*/i + numSurroundingLoops,
                                   &depthSliceUnions[i - 1], strategy);

          if (result.value == FusionResult::Success)
            maxLegalFusionDepth = i;
        }

        if (maxLegalFusionDepth == 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Can't fuse: fusion is not legal at any depth\n");
          continue;
        }

        // Check if fusion would be profitable. We skip profitability analysis
        // for maximal fusion since we already know the maximal legal depth to
        // fuse.
        unsigned bestDstLoopDepth = maxLegalFusionDepth;
        if (!maximalFusion) {
          // Retrieve producer stores from the src loop.
          SmallVector<Operation *, 2> producerStores;
          for (Operation *op : srcNode->stores)
            if (producerConsumerMemrefs.count(
                    cast<AffineWriteOpInterface>(op).getMemRef()))
              producerStores.push_back(op);

          // TODO: Suppport multiple producer stores in profitability
          // analysis. We limit profitability analysis to only scenarios with
          // a single producer store for now. Note that some multi-store
          // producer scenarios will still go through profitability analysis
          // if only one of the stores is involved the producer-consumer
          // relationship of the candidate loops.
          assert(!producerStores.empty() && "Expected producer store");
          if (producerStores.size() > 1)
            LLVM_DEBUG(llvm::dbgs() << "Skipping profitability analysis. Not "
                                       "supported for this case\n");
          else if (!isFusionProfitable(producerStores[0], producerStores[0],
                                       dstAffineForOp, depthSliceUnions,
                                       maxLegalFusionDepth, &bestDstLoopDepth,
                                       computeToleranceThreshold))
            continue;
        }

        assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
        ComputationSliceState &bestSlice =
            depthSliceUnions[bestDstLoopDepth - 1];
        assert(!bestSlice.isEmpty() && "Missing slice union for depth");

        // Determine if 'srcId' can be removed after fusion, taking into
        // account remaining dependences, escaping memrefs and the fusion
        // insertion point.
        bool removeSrcNode = canRemoveSrcNodeAfterFusion(
            srcId, dstId, bestSlice, fusedLoopInsPoint, srcEscapingMemRefs,
            mdg);

        DenseSet<Value> privateMemrefs;
        for (Value memref : producerConsumerMemrefs) {
          if (canCreatePrivateMemRef(memref, srcEscapingMemRefs, srcId, dstId,
                                     removeSrcNode)) {
            // Create a private version of this memref.
            LLVM_DEBUG(llvm::dbgs()
                       << "Creating private memref for " << memref << '\n');
            // Create a private version of this memref.
            privateMemrefs.insert(memref);
          }
        }

        // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
        fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
        dstNodeChanged = true;

        LLVM_DEBUG(llvm::dbgs()
                   << "Fused src loop " << srcId << " into dst loop " << dstId
                   << " at depth " << bestDstLoopDepth << ":\n"
                   << dstAffineForOp << "\n");

        // Move 'dstAffineForOp' before 'insertPointInst' if needed.
        if (fusedLoopInsPoint != dstAffineForOp)
          dstAffineForOp->moveBefore(fusedLoopInsPoint);

        // Update edges between 'srcNode' and 'dstNode'.
        mdg->updateEdges(srcNode->id, dstNode->id, privateMemrefs,
                         removeSrcNode);

        // Create private memrefs.
        if (!privateMemrefs.empty()) {
          // Gather stores for all the private-to-be memrefs.
          DenseMap<Value, SmallVector<Operation *, 4>> privateMemRefToStores;
          dstAffineForOp.walk([&](AffineWriteOpInterface storeOp) {
            Value storeMemRef = storeOp.getMemRef();
            if (privateMemrefs.count(storeMemRef) > 0)
              privateMemRefToStores[storeMemRef].push_back(storeOp);
          });

          // Replace original memrefs with private memrefs. Note that all the
          // loads and stores on these memrefs will be replaced with a new
          // loads and stores. Any reference to the original ones becomes
          // invalid after this point.
          for (auto &memrefToStoresPair : privateMemRefToStores) {
            // TODO: Use union of memref write regions to compute
            // private memref footprint.
            SmallVector<Operation *, 4> &storesForMemref =
                memrefToStoresPair.second;
            Value newMemRef = createPrivateMemRef(
                dstAffineForOp, storesForMemref[0], bestDstLoopDepth,
                fastMemorySpace, localBufSizeThreshold);
            // Create new node in dependence graph for 'newMemRef' alloc op.
            unsigned newMemRefNodeId = mdg->addNode(newMemRef.getDefiningOp());
            // Add edge from 'newMemRef' node to dstNode.
            mdg->addEdge(newMemRefNodeId, dstId, newMemRef);
          }
          // One or more entries for 'newMemRef' alloc op are inserted into
          // the DenseMap mdg->nodes. Since an insertion may cause DenseMap to
          // reallocate, update dstNode.
          dstNode = mdg->getNode(dstId);
        }

        // Collect dst loop stats after memref privatization transformation.
        LoopNestStateCollector dstLoopCollector;
        dstLoopCollector.collect(dstAffineForOp);

        // Clear and add back loads and stores.
        mdg->clearNodeLoadAndStores(dstNode->id);
        mdg->addToNode(dstId, dstLoopCollector.loadOpInsts,
                       dstLoopCollector.storeOpInsts);

        if (removeSrcNode) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Removing src loop " << srcId << " after fusion\n");
          // srcNode is no longer valid after it is removed from mdg.
          srcAffineForOp.erase();
          mdg->removeNode(srcId);
          srcNode = nullptr;
        }
      }
    } while (dstNodeChanged);
  }

  /// Visit each node in the graph, and for each node, attempt to fuse it with
  /// producer-consumer candidates. No fusion is performed when producers with a
  /// user count greater than `maxSrcUserCount` for any of the memrefs involved
  /// are encountered.
  void fuseProducerConsumerNodes(unsigned maxSrcUserCount) {
    LLVM_DEBUG(llvm::dbgs() << "--- Producer/Consumer Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      performFusionsIntoDest(dstId, maxSrcUserCount);
    }
  }

  // Visits each node in the graph, and for each node, attempts to fuse it with
  // its sibling nodes (nodes which share a parent, but no dependence edges).
  void fuseSiblingNodes() {
    LLVM_DEBUG(llvm::dbgs() << "--- Sibling Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Attempt to fuse 'dstNode' with its sibling nodes in the graph.
      fuseWithSiblingNodes(dstNode);
    }
  }

  // Attempt to fuse 'dstNode' with sibling nodes in the graph.
  void fuseWithSiblingNodes(Node *dstNode) {
    DenseSet<unsigned> visitedSibNodeIds;
    std::pair<unsigned, Value> idAndMemref;
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    while (findSiblingNodeToFuse(dstNode, &visitedSibNodeIds, &idAndMemref)) {
      unsigned sibId = idAndMemref.first;
      Value memref = idAndMemref.second;
      // TODO: Check that 'sibStoreOpInst' post-dominates all other
      // stores to the same memref in 'sibNode' loop nest.
      auto *sibNode = mdg->getNode(sibId);
      // Compute an operation list insertion point for the fused loop
      // nest which preserves dependences.
      assert(sibNode->op->getBlock() == dstNode->op->getBlock());
      Operation *insertPointInst =
          sibNode->op->isBeforeInBlock(dstNode->op)
              ? mdg->getFusedLoopNestInsertionPoint(sibNode->id, dstNode->id)
              : mdg->getFusedLoopNestInsertionPoint(dstNode->id, sibNode->id);
      if (insertPointInst == nullptr)
        continue;

      // Check if fusion would be profitable and at what depth.

      // Get unique 'sibNode' load op to 'memref'.
      SmallVector<Operation *, 2> sibLoadOpInsts;
      sibNode->getLoadOpsForMemref(memref, &sibLoadOpInsts);
      // Currently findSiblingNodeToFuse searches for siblings with one load.
      assert(sibLoadOpInsts.size() == 1);
      Operation *sibLoadOpInst = sibLoadOpInsts[0];

      // Gather 'dstNode' load ops to 'memref'.
      SmallVector<Operation *, 2> dstLoadOpInsts;
      dstNode->getLoadOpsForMemref(memref, &dstLoadOpInsts);

      // It's possible this fusion is at an inner depth (i.e., there are common
      // surrounding affine loops for the source and destination for ops). We
      // need to get this number because the call to canFuseLoops needs to be
      // passed the absolute depth. The max legal depth and the depths we try
      // below are however *relative* and as such don't include the common
      // depth.
      SmallVector<AffineForOp, 4> surroundingLoops;
      getAffineForIVs(*dstAffineForOp, &surroundingLoops);
      unsigned numSurroundingLoops = surroundingLoops.size();
      SmallVector<AffineForOp, 4> dstLoopIVs;
      getAffineForIVs(*dstLoadOpInsts[0], &dstLoopIVs);
      unsigned dstLoopDepthTest = dstLoopIVs.size() - numSurroundingLoops;
      auto sibAffineForOp = cast<AffineForOp>(sibNode->op);

      // Compute loop depth and slice union for fusion.
      SmallVector<ComputationSliceState, 8> depthSliceUnions;
      depthSliceUnions.resize(dstLoopDepthTest);
      unsigned maxLegalFusionDepth = 0;
      FusionStrategy strategy(memref);
      for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
        FusionResult result =
            affine::canFuseLoops(sibAffineForOp, dstAffineForOp,
                                 /*dstLoopDepth=*/i + numSurroundingLoops,
                                 &depthSliceUnions[i - 1], strategy);

        if (result.value == FusionResult::Success)
          maxLegalFusionDepth = i;
      }

      LLVM_DEBUG(llvm::dbgs() << "Max legal depth for fusion: "
                              << maxLegalFusionDepth << '\n');

      // Skip if fusion is not feasible at any loop depths.
      if (maxLegalFusionDepth == 0)
        continue;

      unsigned bestDstLoopDepth = maxLegalFusionDepth;
      if (!maximalFusion) {
        // Check if fusion would be profitable. For sibling fusion, the sibling
        // load op is treated as the src "store" op for fusion profitability
        // purposes. The footprint of the load in the slice relative to the
        // unfused source's determines reuse.
        if (!isFusionProfitable(sibLoadOpInst, sibLoadOpInst, dstAffineForOp,
                                depthSliceUnions, maxLegalFusionDepth,
                                &bestDstLoopDepth, computeToleranceThreshold))
          continue;
      }

      assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
      assert(!depthSliceUnions[bestDstLoopDepth - 1].isEmpty() &&
             "Fusion depth has no computed slice union");
      // Check if source loop is being inserted in the innermost
      // destination loop. Based on this, the fused loop may be optimized
      // further inside `fuseLoops`.
      bool isInnermostInsertion = (bestDstLoopDepth == dstLoopDepthTest);
      // Fuse computation slice of 'sibLoopNest' into 'dstLoopNest'.
      affine::fuseLoops(sibAffineForOp, dstAffineForOp,
                        depthSliceUnions[bestDstLoopDepth - 1],
                        isInnermostInsertion);

      auto dstForInst = cast<AffineForOp>(dstNode->op);
      // Update operation position of fused loop nest (if needed).
      if (insertPointInst != dstForInst) {
        dstForInst->moveBefore(insertPointInst);
      }
      // Update data dependence graph state post fusion.
      updateStateAfterSiblingFusion(sibNode, dstNode);
    }
  }

  // Searches block argument uses and the graph from 'dstNode' looking for a
  // fusion candidate sibling node which shares no dependences with 'dstNode'
  // but which loads from the same memref. Returns true and sets
  // 'idAndMemrefToFuse' on success. Returns false otherwise.
  bool findSiblingNodeToFuse(Node *dstNode,
                             DenseSet<unsigned> *visitedSibNodeIds,
                             std::pair<unsigned, Value> *idAndMemrefToFuse) {
    // Returns true if 'sibNode' can be fused with 'dstNode' for input reuse
    // on 'memref'.
    auto canFuseWithSibNode = [&](Node *sibNode, Value memref) {
      // Skip if 'outEdge' is not a read-after-write dependence.
      // TODO: Remove restrict to single load op restriction.
      if (sibNode->getLoadOpCount(memref) != 1)
        return false;
      // Skip if there exists a path of dependent edges between
      // 'sibNode' and 'dstNode'.
      if (mdg->hasDependencePath(sibNode->id, dstNode->id) ||
          mdg->hasDependencePath(dstNode->id, sibNode->id))
        return false;
      // Skip sib node if it loads to (and stores from) the same memref on
      // which it also has an input dependence edge.
      DenseSet<Value> loadAndStoreMemrefSet;
      sibNode->getLoadAndStoreMemrefSet(&loadAndStoreMemrefSet);
      if (llvm::any_of(loadAndStoreMemrefSet, [=](Value memref) {
            return mdg->getIncomingMemRefAccesses(sibNode->id, memref) > 0;
          }))
        return false;

      // Check that all stores are to the same memref if any.
      DenseSet<Value> storeMemrefs;
      for (auto *storeOpInst : sibNode->stores) {
        storeMemrefs.insert(
            cast<AffineWriteOpInterface>(storeOpInst).getMemRef());
      }
      if (storeMemrefs.size() > 1)
        return false;

      // Skip if a memref value in one node is used by a non-affine memref
      // access that lies between 'dstNode' and 'sibNode'.
      if (hasNonAffineUsersOnThePath(dstNode->id, sibNode->id, mdg) ||
          hasNonAffineUsersOnThePath(sibNode->id, dstNode->id, mdg))
        return false;
      return true;
    };

    // Search for siblings which load the same memref block argument.
    Block *block = dstNode->op->getBlock();
    for (unsigned i = 0, e = block->getNumArguments(); i != e; ++i) {
      for (Operation *user : block->getArgument(i).getUsers()) {
        auto loadOp = dyn_cast<AffineReadOpInterface>(user);
        if (!loadOp)
          continue;
        // Gather loops surrounding 'use'.
        SmallVector<AffineForOp, 4> loops;
        getAffineForIVs(*user, &loops);
        // Skip 'use' if it is not within a loop nest.
        // Find the surrounding affine.for nested immediately within the
        // block.
        auto *it = llvm::find_if(loops, [&](AffineForOp loop) {
          return loop->getBlock() == &mdg->block;
        });
        // Skip 'use' if it is not within a loop nest in `block`.
        if (it == loops.end())
          continue;
        Node *sibNode = mdg->getForOpNode(*it);
        assert(sibNode != nullptr);
        // Skip 'use' if it not a sibling to 'dstNode'.
        if (sibNode->id == dstNode->id)
          continue;
        // Skip 'use' if it has been visited.
        if (visitedSibNodeIds->count(sibNode->id) > 0)
          continue;
        // Skip 'use' if it does not load from the same memref as 'dstNode'.
        auto memref = loadOp.getMemRef();
        if (dstNode->getLoadOpCount(memref) == 0)
          continue;
        // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
        if (canFuseWithSibNode(sibNode, memref)) {
          visitedSibNodeIds->insert(sibNode->id);
          idAndMemrefToFuse->first = sibNode->id;
          idAndMemrefToFuse->second = memref;
          return true;
        }
      }
    }

    // Search for siblings by following edges through an intermediate src node.
    // Collect candidate 'dstNode' input edges in 'inEdges'.
    SmallVector<MemRefDependenceGraph::Edge, 2> inEdges;
    mdg->forEachMemRefInputEdge(
        dstNode->id, [&](MemRefDependenceGraph::Edge inEdge) {
          // Add 'inEdge' if it is a read-after-write dependence.
          if (dstNode->getLoadOpCount(inEdge.value) > 0 &&
              mdg->getNode(inEdge.id)->getStoreOpCount(inEdge.value) > 0)
            inEdges.push_back(inEdge);
        });

    // Search for sibling nodes to fuse by visiting output edges from each input
    // edge in 'inEdges'.
    for (auto &inEdge : inEdges) {
      // Collect candidate output edges from each node 'inEdge.id' in 'inEdges'.
      SmallVector<MemRefDependenceGraph::Edge, 2> outEdges;
      mdg->forEachMemRefOutputEdge(
          inEdge.id, [&](MemRefDependenceGraph::Edge outEdge) {
            unsigned sibNodeId = outEdge.id;
            if (visitedSibNodeIds->count(sibNodeId) > 0)
              return;
            // Skip output edge if not a sibling using the same memref.
            if (outEdge.id == dstNode->id || outEdge.value != inEdge.value)
              return;
            auto *sibNode = mdg->getNode(sibNodeId);
            if (!isa<AffineForOp>(sibNode->op))
              return;
            // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
            if (canFuseWithSibNode(sibNode, outEdge.value)) {
              // Add candidate 'outEdge' to sibling node.
              outEdges.push_back(outEdge);
            }
          });

      // Add first candidate if any were returned.
      if (!outEdges.empty()) {
        visitedSibNodeIds->insert(outEdges[0].id);
        idAndMemrefToFuse->first = outEdges[0].id;
        idAndMemrefToFuse->second = outEdges[0].value;
        return true;
      }
    }
    return false;
  }

  /// Update data dependence graph state to reflect sibling fusion of 'sibNode'
  /// into 'dstNode'.
  void updateStateAfterSiblingFusion(Node *sibNode, Node *dstNode) {
    // Update 'sibNode' and 'dstNode' input/output edges to reflect fusion.
    mdg->updateEdges(sibNode->id, dstNode->id);

    // Collect dst loop stats after memref privatization transformation.
    auto dstForInst = cast<AffineForOp>(dstNode->op);
    LoopNestStateCollector dstLoopCollector;
    dstLoopCollector.collect(dstForInst);
    // Clear and add back loads and stores
    mdg->clearNodeLoadAndStores(dstNode->id);
    mdg->addToNode(dstNode->id, dstLoopCollector.loadOpInsts,
                   dstLoopCollector.storeOpInsts);
    // Remove old sibling loop nest if it no longer has outgoing dependence
    // edges, and it does not write to a memref which escapes the block.
    if (mdg->getOutEdgeCount(sibNode->id) == 0) {
      Operation *op = sibNode->op;
      mdg->removeNode(sibNode->id);
      op->erase();
    }
  }

  // Clean up any allocs with no users.
  void eraseUnusedMemRefAllocations() {
    for (auto &pair : mdg->memrefEdgeCount) {
      if (pair.second > 0)
        continue;
      auto memref = pair.first;
      // Skip if there exist other uses (return operation or function calls).
      if (!memref.use_empty())
        continue;
      // Use list expected to match the dep graph info.
      auto *op = memref.getDefiningOp();
      if (isa_and_nonnull<memref::AllocOp>(op))
        op->erase();
    }
  }
};

} // namespace

/// Run fusion on `block`.
void LoopFusion::runOnBlock(Block *block) {
  MemRefDependenceGraph g(*block);
  if (!g.init()) {
    LLVM_DEBUG(llvm::dbgs() << "MDG init failed\n");
    return;
  }

  std::optional<unsigned> fastMemorySpaceOpt;
  if (fastMemorySpace.hasValue())
    fastMemorySpaceOpt = fastMemorySpace;
  unsigned localBufSizeThresholdBytes = localBufSizeThreshold * 1024;
  GreedyFusion fusion(&g, localBufSizeThresholdBytes, fastMemorySpaceOpt,
                      maximalFusion, computeToleranceThreshold);

  if (affineFusionMode == FusionMode::ProducerConsumer)
    fusion.runProducerConsumerFusionOnly();
  else if (affineFusionMode == FusionMode::Sibling)
    fusion.runSiblingFusionOnly();
  else if (affineFusionMode == FusionMode::Independent)  // 新增
    fusion.runIndependentFusion();
  else
    fusion.runGreedyFusion();
}

void LoopFusion::runOnOperation() {
  // Call fusion on every op that has at least two affine.for nests (in post
  // order).
  getOperation()->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        auto affineFors = block.getOps<AffineForOp>();
        if (!affineFors.empty() && !llvm::hasSingleElement(affineFors))
          runOnBlock(&block);
      }
    }
  });
}

std::unique_ptr<Pass> mlir::affine::createLoopFusionPass(
    unsigned fastMemorySpace, uint64_t localBufSizeThreshold,
    bool maximalFusion, enum FusionMode affineFusionMode) {
  return std::make_unique<LoopFusion>(fastMemorySpace, localBufSizeThreshold,
                                      maximalFusion, affineFusionMode);
}


