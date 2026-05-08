//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct ConvertAlloca final : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  // LogicalResult
  // matchAndRewrite(memref::AllocaOp op, OpAdaptor operands,
  //                 ConversionPatternRewriter &rewriter) const override {

  //   if (!op.getType().hasStaticShape()) {
  //     return rewriter.notifyMatchFailure(
  //         op.getLoc(), "cannot transform alloca with dynamic shape");
  //   }

  //   // if (op.getAlignment().value_or(1) > 1) {
  //   //   // TODO: Allow alignment if it is not more than the natural alignment
  //   //   // of the C array.
  //   //   return rewriter.notifyMatchFailure(
  //   //       op.getLoc(), "cannot transform alloca with alignment requirement");
  //   // }

  //   auto resultTy = getTypeConverter()->convertType(op.getType());
  //   if (!resultTy) {
  //     return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
  //   }
  //   auto noInit = emitc::OpaqueAttr::get(getContext(), "");
  //   rewriter.replaceOpWithNewOp<emitc::VariableOp>(op, resultTy, noInit);
  //   return success();
  // }

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefType = op.getType();
    if (!memrefType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with dynamic shape");
    }

    // 核心修改：如果是 0-rank，升维成 1-rank 进行类型转换
    Type typeToConvert = memrefType;
    if (memrefType.getRank() == 0) {
      typeToConvert = MemRefType::get({1}, memrefType.getElementType());
    }

    auto resultTy = getTypeConverter()->convertType(typeToConvert);
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(op, resultTy, noInit);
    return success();
  }

};

struct ConvertGlobal final : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform global with dynamic shape");
    }

    // if (op.getAlignment().value_or(1) > 1) {
    //   // TODO: Extend GlobalOp to specify alignment via the `alignas` specifier.
    //   return rewriter.notifyMatchFailure(
    //       op.getLoc(), "global variable with alignment requirement is "
    //                    "currently not supported");
    // }
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }

    SymbolTable::Visibility visibility = SymbolTable::getSymbolVisibility(op);
    if (visibility != SymbolTable::Visibility::Public &&
        visibility != SymbolTable::Visibility::Private) {
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "only public and private visibility is currently supported");
    }
    // We are explicit in specifing the linkage because the default linkage
    // for constants is different in C and C++.
    bool staticSpecifier = visibility == SymbolTable::Visibility::Private;
    bool externSpecifier = !staticSpecifier;

    Attribute initialValue = operands.getInitialValueAttr();
    if (isa_and_present<UnitAttr>(initialValue))
      initialValue = {};

    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, operands.getSymName(), resultTy, initialValue, externSpecifier,
        staticSpecifier, operands.getConstant());
    return success();
  }
};

struct ConvertGetGlobal final
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }
    rewriter.replaceOpWithNewOp<emitc::GetGlobalOp>(op, resultTy,
                                                    operands.getNameAttr());
    return success();
  }
};

struct ConvertLoad final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  // LogicalResult
  // matchAndRewrite(memref::LoadOp op, OpAdaptor operands,
  //                 ConversionPatternRewriter &rewriter) const override {

  //   auto resultTy = getTypeConverter()->convertType(op.getType());
  //   if (!resultTy) {
  //     return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
  //   }

  //   auto arrayValue =
  //       dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
  //   if (!arrayValue) {
  //     return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
  //   }

  //   auto subscript = rewriter.create<emitc::SubscriptOp>(
  //       op.getLoc(), arrayValue, operands.getIndices());

  //   auto noInit = emitc::OpaqueAttr::get(getContext(), "");
  //   auto var =
  //       rewriter.create<emitc::VariableOp>(op.getLoc(), resultTy, noInit);

  //   rewriter.create<emitc::AssignOp>(op.getLoc(), var, subscript);
  //   rewriter.replaceOp(op, var);
  //   return success();
  // }

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
    if (!arrayValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    // 核心修改：如果是 0-rank，补齐索引 [0]
    ValueRange indices = operands.getIndices();
    SmallVector<Value, 1> newIndices;
    if (op.getMemRefType().getRank() == 0) {
      auto indexTy = rewriter.getIndexType();
      auto zero = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), indexTy, rewriter.getIndexAttr(0));
      newIndices.push_back(zero);
      indices = newIndices;
    }

    auto subscript = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), arrayValue, indices);

    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    auto var =
        rewriter.create<emitc::VariableOp>(op.getLoc(), resultTy, noInit);

    rewriter.create<emitc::AssignOp>(op.getLoc(), var, subscript);
    rewriter.replaceOp(op, var);
    return success();
  }


};

struct ConvertStore final : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  // LogicalResult
  // matchAndRewrite(memref::StoreOp op, OpAdaptor operands,
  //                 ConversionPatternRewriter &rewriter) const override {
  //   auto arrayValue =
  //       dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
  //   if (!arrayValue) {
  //     return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
  //   }

  //   auto subscript = rewriter.create<emitc::SubscriptOp>(
  //       op.getLoc(), arrayValue, operands.getIndices());
  //   rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscript,
  //                                                operands.getValue());
  //   return success();
  // }


  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {             
    // llvm::outs() << "Converting StoreOp, Rank: ";
    // 获取原始的 MemRef 类型以判断 Rank
    auto memrefType = op.getMemRefType();
    Location loc = op.getLoc();

    // 1. 获取转换后的 MemRef 操作数
    Value memrefValue = operands.getMemref();
    
    // 2. 处理索引
    ValueRange indices = operands.getIndices();
    SmallVector<Value, 1> newIndices;

    // 如果原始是 0-rank，我们需要手动补上索引 [0]
    if (memrefType.getRank() == 0) {
      auto indexTy = rewriter.getIndexType();
      // 创建常量 0
      auto zero = rewriter.create<emitc::ConstantOp>(
          loc, indexTy, rewriter.getIndexAttr(0));
      newIndices.push_back(zero);
      indices = newIndices;
    }

    // 3. 创建 Subscript 操作
    // 注意：这里直接使用元素的类型。如果是 memref<i1>，元素类型就是 i1
    auto subscript = rewriter.create<emitc::SubscriptOp>(
        loc, memrefType.getElementType(), memrefValue, indices);

    // 4. 使用 AssignOp 完成赋值
    // 对应 C++: ptr[0] = value;
    rewriter.replaceOpWithNewOp<emitc::AssignOp>(op, subscript, operands.getValue());
    
    return success();
  }

};
} // namespace

// void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
//   typeConverter.addConversion(
//       [&](MemRefType memRefType) -> std::optional<Type> {
//         if (!memRefType.hasStaticShape() ||
//             !memRefType.getLayout().isIdentity() || memRefType.getRank() == 0) {
//           return {};
//         }
//         Type convertedElementType =
//             typeConverter.convertType(memRefType.getElementType());
//         if (!convertedElementType)
//           return {};
//         return emitc::ArrayType::get(memRefType.getShape(),
//                                      convertedElementType);
//       });
// }

struct FoldExtractCastToEmitCApplyAddr final
    : public OpConversionPattern<memref::ExtractAlignedPointerAsIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Require exactly one user: the emitc.cast.
    if (!op->hasOneUse())
      return rewriter.notifyMatchFailure(loc, "extract does not have exactly one use");

    auto cast = dyn_cast<emitc::CastOp>(*op->getUsers().begin());
    if (!cast)
      return rewriter.notifyMatchFailure(loc, "extract's user is not emitc.cast");

    // cast must produce !emitc.ptr<...>
    auto dstPtrTy = dyn_cast<emitc::PointerType>(cast.getType());
    if (!dstPtrTy)
      return rewriter.notifyMatchFailure(loc, "cast result is not !emitc.ptr");

    // In dialect conversion, adaptor.getSource() is the converted operand of extract,
    // expected to be !emitc.array<...>.
    Value convertedSrc = adaptor.getSource();
    if (!isa<emitc::ArrayType>(convertedSrc.getType()))
      return rewriter.notifyMatchFailure(loc, "extract source not converted to !emitc.array");

    // Build: %p = emitc.apply "&"(%convertedSrc) : (!emitc.array<...>) -> !emitc.ptr<...>
    // NOTE: ApplyOp takes a single Value operand (not ValueRange).
    auto addr = rewriter.create<emitc::ApplyOp>(
        loc,
        /*result=*/dstPtrTy,
        // /*applicableOperator=*/rewriter.getStringAttr("&"),
        /*applicableOperator=*/rewriter.getStringAttr("(float*)"),
        /*operand=*/convertedSrc);

    // Replace the cast (not the extract) with the apply result, then erase extract.
    rewriter.replaceOp(cast, addr.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](MemRefType memRefType) -> std::optional<Type> {
        // 1. 依然不支持动态形状和非 Identity 布局
        if (!memRefType.hasStaticShape() || !memRefType.getLayout().isIdentity()) {
          return {};
        }

        // 2. 转换元素类型
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType)
          return {};

        // 3. 核心修改：如果是 0-rank，返回长度为 1 的数组类型
        if (memRefType.getRank() == 0) {
          return emitc::ArrayType::get({1}, convertedElementType);
        }

        // 4. 普通多维情况
        return emitc::ArrayType::get(memRefType.getShape(),
                                     convertedElementType);
      });
}

void mlir::populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                   TypeConverter &converter) {
  patterns.add<ConvertAlloca, ConvertGlobal, ConvertGetGlobal, ConvertLoad,
               ConvertStore, FoldExtractCastToEmitCApplyAddr>(converter, patterns.getContext());
}
