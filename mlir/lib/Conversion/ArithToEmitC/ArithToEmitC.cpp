// //===- ArithToEmitC.cpp - Arith to EmitC Patterns ---------------*- C++ -*-===//
// //
// // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// // See https://llvm.org/LICENSE.txt for license information.
// // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// //
// //===----------------------------------------------------------------------===//
// //
// // This file implements patterns to convert the Arith dialect to the EmitC
// // dialect.
// //
// //===----------------------------------------------------------------------===//

// #include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"

// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/EmitC/IR/EmitC.h"
// #include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinTypes.h"
// #include "mlir/Transforms/DialectConversion.h"

// using namespace mlir;

// //===----------------------------------------------------------------------===//
// // Conversion Patterns
// //===----------------------------------------------------------------------===//

// namespace {
// class ArithConstantOpConversionPattern
//     : public OpConversionPattern<arith::ConstantOp> {
// public:
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(arith::ConstantOp arithConst,
//                   arith::ConstantOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     Type newTy = this->getTypeConverter()->convertType(arithConst.getType());
//     if (!newTy)
//       return rewriter.notifyMatchFailure(arithConst, "type conversion failed");
//     rewriter.replaceOpWithNewOp<emitc::ConstantOp>(arithConst, newTy,
//                                                    adaptor.getValue());
//     return success();
//   }
// };



// /// Get the signed or unsigned type corresponding to \p ty.
// Type adaptIntegralTypeSignedness(Type ty, bool needsUnsigned) {
//   if (isa<IntegerType>(ty)) {
//     if (ty.isUnsignedInteger() != needsUnsigned) {
//       auto signedness = needsUnsigned
//                             ? IntegerType::SignednessSemantics::Unsigned
//                             : IntegerType::SignednessSemantics::Signed;
//       return IntegerType::get(ty.getContext(), ty.getIntOrFloatBitWidth(),
//                               signedness);
//     }
//   } else if (emitc::isPointerWideType(ty)) {
//     if (isa<emitc::SizeTType>(ty) != needsUnsigned) {
//       if (needsUnsigned)
//         return emitc::SizeTType::get(ty.getContext());
//       return emitc::PtrDiffTType::get(ty.getContext());
//     }
//   }
//   return ty;
// }

// /// Insert a cast operation to type \p ty if \p val does not have this type.
// Value adaptValueType(Value val, ConversionPatternRewriter &rewriter, Type ty) {
//   return rewriter.createOrFold<emitc::CastOp>(val.getLoc(), ty, val);
// }

// class CmpFOpConversion : public OpConversionPattern<arith::CmpFOp> {
// public:
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     if (!isa<FloatType>(adaptor.getRhs().getType())) {
//       return rewriter.notifyMatchFailure(op.getLoc(),
//                                          "cmpf currently only supported on "
//                                          "floats, not tensors/vectors thereof");
//     }

//     bool unordered = false;
//     emitc::CmpPredicate predicate;
//     switch (op.getPredicate()) {
//     case arith::CmpFPredicate::AlwaysFalse: {
//       auto constant = rewriter.create<emitc::ConstantOp>(
//           op.getLoc(), rewriter.getI1Type(),
//           rewriter.getBoolAttr(/*value=*/false));
//       rewriter.replaceOp(op, constant);
//       return success();
//     }
//     case arith::CmpFPredicate::OEQ:
//       unordered = false;
//       predicate = emitc::CmpPredicate::eq;
//       break;
//     case arith::CmpFPredicate::OGT:
//       unordered = false;
//       predicate = emitc::CmpPredicate::gt;
//       break;
//     case arith::CmpFPredicate::OGE:
//       unordered = false;
//       predicate = emitc::CmpPredicate::ge;
//       break;
//     case arith::CmpFPredicate::OLT:
//       unordered = false;
//       predicate = emitc::CmpPredicate::lt;
//       break;
//     case arith::CmpFPredicate::OLE:
//       unordered = false;
//       predicate = emitc::CmpPredicate::le;
//       break;
//     case arith::CmpFPredicate::ONE:
//       unordered = false;
//       predicate = emitc::CmpPredicate::ne;
//       break;
//     case arith::CmpFPredicate::ORD: {
//       // ordered, i.e. none of the operands is NaN
//       auto cmp = createCheckIsOrdered(rewriter, op.getLoc(), adaptor.getLhs(),
//                                       adaptor.getRhs());
//       rewriter.replaceOp(op, cmp);
//       return success();
//     }
//     case arith::CmpFPredicate::UEQ:
//       unordered = true;
//       predicate = emitc::CmpPredicate::eq;
//       break;
//     case arith::CmpFPredicate::UGT:
//       unordered = true;
//       predicate = emitc::CmpPredicate::gt;
//       break;
//     case arith::CmpFPredicate::UGE:
//       unordered = true;
//       predicate = emitc::CmpPredicate::ge;
//       break;
//     case arith::CmpFPredicate::ULT:
//       unordered = true;
//       predicate = emitc::CmpPredicate::lt;
//       break;
//     case arith::CmpFPredicate::ULE:
//       unordered = true;
//       predicate = emitc::CmpPredicate::le;
//       break;
//     case arith::CmpFPredicate::UNE:
//       unordered = true;
//       predicate = emitc::CmpPredicate::ne;
//       break;
//     case arith::CmpFPredicate::UNO: {
//       // unordered, i.e. either operand is nan
//       auto cmp = createCheckIsUnordered(rewriter, op.getLoc(), adaptor.getLhs(),
//                                         adaptor.getRhs());
//       rewriter.replaceOp(op, cmp);
//       return success();
//     }
//     case arith::CmpFPredicate::AlwaysTrue: {
//       auto constant = rewriter.create<emitc::ConstantOp>(
//           op.getLoc(), rewriter.getI1Type(),
//           rewriter.getBoolAttr(/*value=*/true));
//       rewriter.replaceOp(op, constant);
//       return success();
//     }
//     }

//     // Compare the values naively
//     auto cmpResult =
//         rewriter.create<emitc::CmpOp>(op.getLoc(), op.getType(), predicate,
//                                       adaptor.getLhs(), adaptor.getRhs());

//     // Adjust the results for unordered/ordered semantics
//     if (unordered) {
//       auto isUnordered = createCheckIsUnordered(
//           rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
//       rewriter.replaceOpWithNewOp<emitc::LogicalOrOp>(op, op.getType(),
//                                                       isUnordered, cmpResult);
//       return success();
//     }

//     auto isOrdered = createCheckIsOrdered(rewriter, op.getLoc(),
//                                           adaptor.getLhs(), adaptor.getRhs());
//     rewriter.replaceOpWithNewOp<emitc::LogicalAndOp>(op, op.getType(),
//                                                      isOrdered, cmpResult);
//     return success();
//   }

// private:
//   /// Return a value that is true if \p operand is NaN.
//   Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
//               Value operand) const {
//     // A value is NaN exactly when it compares unequal to itself.
//     return rewriter.create<emitc::CmpOp>(
//         loc, rewriter.getI1Type(), emitc::CmpPredicate::ne, operand, operand);
//   }

//   /// Return a value that is true if \p operand is not NaN.
//   Value isNotNaN(ConversionPatternRewriter &rewriter, Location loc,
//                  Value operand) const {
//     // A value is not NaN exactly when it compares equal to itself.
//     return rewriter.create<emitc::CmpOp>(
//         loc, rewriter.getI1Type(), emitc::CmpPredicate::eq, operand, operand);
//   }

//   /// Return a value that is true if the operands \p first and \p second are
//   /// unordered (i.e., at least one of them is NaN).
//   Value createCheckIsUnordered(ConversionPatternRewriter &rewriter,
//                                Location loc, Value first, Value second) const {
//     auto firstIsNaN = isNaN(rewriter, loc, first);
//     auto secondIsNaN = isNaN(rewriter, loc, second);
//     return rewriter.create<emitc::LogicalOrOp>(loc, rewriter.getI1Type(),
//                                                firstIsNaN, secondIsNaN);
//   }

//   /// Return a value that is true if the operands \p first and \p second are
//   /// both ordered (i.e., none one of them is NaN).
//   Value createCheckIsOrdered(ConversionPatternRewriter &rewriter, Location loc,
//                              Value first, Value second) const {
//     auto firstIsNotNaN = isNotNaN(rewriter, loc, first);
//     auto secondIsNotNaN = isNotNaN(rewriter, loc, second);
//     return rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
//                                                 firstIsNotNaN, secondIsNotNaN);
//   }
// };

// class CmpIOpConversion : public OpConversionPattern<arith::CmpIOp> {
// public:
//   using OpConversionPattern::OpConversionPattern;

//   bool needsUnsignedCmp(arith::CmpIPredicate pred) const {
//     switch (pred) {
//     case arith::CmpIPredicate::eq:
//     case arith::CmpIPredicate::ne:
//     case arith::CmpIPredicate::slt:
//     case arith::CmpIPredicate::sle:
//     case arith::CmpIPredicate::sgt:
//     case arith::CmpIPredicate::sge:
//       return false;
//     case arith::CmpIPredicate::ult:
//     case arith::CmpIPredicate::ule:
//     case arith::CmpIPredicate::ugt:
//     case arith::CmpIPredicate::uge:
//       return true;
//     }
//     llvm_unreachable("unknown cmpi predicate kind");
//   }

//   emitc::CmpPredicate toEmitCPred(arith::CmpIPredicate pred) const {
//     switch (pred) {
//     case arith::CmpIPredicate::eq:
//       return emitc::CmpPredicate::eq;
//     case arith::CmpIPredicate::ne:
//       return emitc::CmpPredicate::ne;
//     case arith::CmpIPredicate::slt:
//     case arith::CmpIPredicate::ult:
//       return emitc::CmpPredicate::lt;
//     case arith::CmpIPredicate::sle:
//     case arith::CmpIPredicate::ule:
//       return emitc::CmpPredicate::le;
//     case arith::CmpIPredicate::sgt:
//     case arith::CmpIPredicate::ugt:
//       return emitc::CmpPredicate::gt;
//     case arith::CmpIPredicate::sge:
//     case arith::CmpIPredicate::uge:
//       return emitc::CmpPredicate::ge;
//     }
//     llvm_unreachable("unknown cmpi predicate kind");
//   }

//   LogicalResult
//   matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type type = adaptor.getLhs().getType();
//     if (!type || !(isa<IntegerType>(type) || emitc::isPointerWideType(type))) {
//       return rewriter.notifyMatchFailure(
//           op, "expected integer or size_t/ssize_t/ptrdiff_t type");
//     }

//     bool needsUnsigned = needsUnsignedCmp(op.getPredicate());
//     emitc::CmpPredicate pred = toEmitCPred(op.getPredicate());

//     Type arithmeticType = adaptIntegralTypeSignedness(type, needsUnsigned);
//     Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
//     Value rhs = adaptValueType(adaptor.getRhs(), rewriter, arithmeticType);

//     rewriter.replaceOpWithNewOp<emitc::CmpOp>(op, op.getType(), pred, lhs, rhs);
//     return success();
//   }
// };

// class NegFOpConversion : public OpConversionPattern<arith::NegFOp> {
// public:
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     auto adaptedOp = adaptor.getOperand();
//     auto adaptedOpType = adaptedOp.getType();

//     if (isa<TensorType>(adaptedOpType) || isa<VectorType>(adaptedOpType)) {
//       return rewriter.notifyMatchFailure(
//           op.getLoc(),
//           "negf currently only supports scalar types, not vectors or tensors");
//     }

//     if (!emitc::isSupportedFloatType(adaptedOpType)) {
//       return rewriter.notifyMatchFailure(
//           op.getLoc(), "floating-point type is not supported by EmitC");
//     }

//     rewriter.replaceOpWithNewOp<emitc::UnaryMinusOp>(op, adaptedOpType,
//                                                      adaptedOp);
//     return success();
//   }
// };

// template <typename ArithOp, bool castToUnsigned>
// class CastConversion : public OpConversionPattern<ArithOp> {
// public:
//   using OpConversionPattern<ArithOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type opReturnType = this->getTypeConverter()->convertType(op.getType());
//     if (!opReturnType || !(isa<IntegerType>(opReturnType) ||
//                            emitc::isPointerWideType(opReturnType)))
//       return rewriter.notifyMatchFailure(
//           op, "expected integer or size_t/ssize_t/ptrdiff_t result type");

//     if (adaptor.getOperands().size() != 1) {
//       return rewriter.notifyMatchFailure(
//           op, "CastConversion only supports unary ops");
//     }

//     Type operandType = adaptor.getIn().getType();
//     if (!operandType || !(isa<IntegerType>(operandType) ||
//                           emitc::isPointerWideType(operandType)))
//       return rewriter.notifyMatchFailure(
//           op, "expected integer or size_t/ssize_t/ptrdiff_t operand type");

//     // Signed (sign-extending) casts from i1 are not supported.
//     if (operandType.isInteger(1) && !castToUnsigned)
//       return rewriter.notifyMatchFailure(op,
//                                          "operation not supported on i1 type");

//     // to-i1 conversions: arith semantics want truncation, whereas (bool)(v) is
//     // equivalent to (v != 0). Implementing as (bool)(v & 0x01) gives
//     // truncation.
//     if (opReturnType.isInteger(1)) {
//       Type attrType = (emitc::isPointerWideType(operandType))
//                           ? rewriter.getIndexType()
//                           : operandType;
//       auto constOne = rewriter.create<emitc::ConstantOp>(
//           op.getLoc(), operandType, rewriter.getOneAttr(attrType));
//       auto oneAndOperand = rewriter.create<emitc::BitwiseAndOp>(
//           op.getLoc(), operandType, adaptor.getIn(), constOne);
//       rewriter.replaceOpWithNewOp<emitc::CastOp>(op, opReturnType,
//                                                  oneAndOperand);
//       return success();
//     }

//     bool isTruncation =
//         (isa<IntegerType>(operandType) && isa<IntegerType>(opReturnType) &&
//          operandType.getIntOrFloatBitWidth() >
//              opReturnType.getIntOrFloatBitWidth());
//     bool doUnsigned = castToUnsigned || isTruncation;

//     // Adapt the signedness of the result (bitwidth-preserving cast)
//     // This is needed e.g., if the return type is signless.
//     Type castDestType = adaptIntegralTypeSignedness(opReturnType, doUnsigned);

//     // Adapt the signedness of the operand (bitwidth-preserving cast)
//     Type castSrcType = adaptIntegralTypeSignedness(operandType, doUnsigned);
//     Value actualOp = adaptValueType(adaptor.getIn(), rewriter, castSrcType);

//     // Actual cast (may change bitwidth)
//     auto cast = rewriter.template create<emitc::CastOp>(op.getLoc(),
//                                                         castDestType, actualOp);

//     // Cast to the expected output type
//     auto result = adaptValueType(cast, rewriter, opReturnType);

//     rewriter.replaceOp(op, result);
//     return success();
//   }
// };

// template <typename ArithOp>
// class UnsignedCastConversion : public CastConversion<ArithOp, true> {
//   using CastConversion<ArithOp, true>::CastConversion;
// };

// template <typename ArithOp>
// class SignedCastConversion : public CastConversion<ArithOp, false> {
//   using CastConversion<ArithOp, false>::CastConversion;
// };

// template <typename ArithOp, typename EmitCOp>
// class ArithOpConversion final : public OpConversionPattern<ArithOp> {
// public:
//   using OpConversionPattern<ArithOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ArithOp arithOp, typename ArithOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type newTy = this->getTypeConverter()->convertType(arithOp.getType());
//     if (!newTy)
//       return rewriter.notifyMatchFailure(arithOp,
//                                          "converting result type failed");
//     rewriter.template replaceOpWithNewOp<EmitCOp>(arithOp, newTy,
//                                                   adaptor.getOperands());

//     return success();
//   }
// };

// template <typename ArithOp, typename EmitCOp>
// class IntegerOpConversion final : public OpConversionPattern<ArithOp> {
// public:
//   using OpConversionPattern<ArithOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type type = this->getTypeConverter()->convertType(op.getType());
//     if (!type || !(isa<IntegerType>(type) || emitc::isPointerWideType(type))) {
//       return rewriter.notifyMatchFailure(
//           op, "expected integer or size_t/ssize_t/ptrdiff_t type");
//     }

//     if (type.isInteger(1)) {
//       // arith expects wrap-around arithmethic, which doesn't happen on `bool`.
//       return rewriter.notifyMatchFailure(op, "i1 type is not implemented");
//     }

//     Type arithmeticType = type;
//     if ((type.isSignlessInteger() || type.isSignedInteger()) &&
//         !bitEnumContainsAll(op.getOverflowFlags(),
//                             arith::IntegerOverflowFlags::nsw)) {
//       // If the C type is signed and the op doesn't guarantee "No Signed Wrap",
//       // we compute in unsigned integers to avoid UB.
//       arithmeticType = rewriter.getIntegerType(type.getIntOrFloatBitWidth(),
//                                                /*isSigned=*/false);
//     }

//     Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
//     Value rhs = adaptValueType(adaptor.getRhs(), rewriter, arithmeticType);

//     Value arithmeticResult = rewriter.template create<EmitCOp>(
//         op.getLoc(), arithmeticType, lhs, rhs);

//     Value result = adaptValueType(arithmeticResult, rewriter, type);

//     rewriter.replaceOp(op, result);
//     return success();
//   }
// };

// template <typename ArithOp, typename EmitCOp>
// class BitwiseOpConversion : public OpConversionPattern<ArithOp> {
// public:
//   using OpConversionPattern<ArithOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type type = this->getTypeConverter()->convertType(op.getType());
//     if (!isa_and_nonnull<IntegerType>(type)) {
//       return rewriter.notifyMatchFailure(
//           op,
//           "expected integer type, vector/tensor support not yet implemented");
//     }

//     // Bitwise ops can be performed directly on booleans
//     if (type.isInteger(1)) {
//       rewriter.replaceOpWithNewOp<EmitCOp>(op, type, adaptor.getLhs(),
//                                            adaptor.getRhs());
//       return success();
//     }

//     // Bitwise ops are defined by the C standard on unsigned operands.
//     Type arithmeticType =
//         adaptIntegralTypeSignedness(type, /*needsUnsigned=*/true);

//     Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
//     Value rhs = adaptValueType(adaptor.getRhs(), rewriter, arithmeticType);

//     Value arithmeticResult = rewriter.template create<EmitCOp>(
//         op.getLoc(), arithmeticType, lhs, rhs);

//     Value result = adaptValueType(arithmeticResult, rewriter, type);

//     rewriter.replaceOp(op, result);
//     return success();
//   }
// };

// template <typename ArithOp, typename EmitCOp, bool isUnsignedOp>
// class ShiftOpConversion : public OpConversionPattern<ArithOp> {
// public:
//   using OpConversionPattern<ArithOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type type = this->getTypeConverter()->convertType(op.getType());
//     if (!type || !(isa<IntegerType>(type) || emitc::isPointerWideType(type))) {
//       return rewriter.notifyMatchFailure(
//           op, "expected integer or size_t/ssize_t/ptrdiff_t type");
//     }

//     if (type.isInteger(1)) {
//       return rewriter.notifyMatchFailure(op, "i1 type is not implemented");
//     }

//     Type arithmeticType = adaptIntegralTypeSignedness(type, isUnsignedOp);

//     Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
//     // Shift amount interpreted as unsigned per Arith dialect spec.
//     Type rhsType = adaptIntegralTypeSignedness(adaptor.getRhs().getType(),
//                                                /*needsUnsigned=*/true);
//     Value rhs = adaptValueType(adaptor.getRhs(), rewriter, rhsType);

//     // Add a runtime check for overflow
//     Value width;
//     if (emitc::isPointerWideType(type)) {
//       Value eight = rewriter.create<emitc::ConstantOp>(
//           op.getLoc(), rhsType, rewriter.getIndexAttr(8));
//       emitc::CallOpaqueOp sizeOfCall = rewriter.create<emitc::CallOpaqueOp>(
//           op.getLoc(), rhsType, "sizeof", ArrayRef<Value>{eight});
//       width = rewriter.create<emitc::MulOp>(op.getLoc(), rhsType, eight,
//                                             sizeOfCall.getResult(0));
//     } else {
//       width = rewriter.create<emitc::ConstantOp>(
//           op.getLoc(), rhsType,
//           rewriter.getIntegerAttr(rhsType, type.getIntOrFloatBitWidth()));
//     }

//     Value excessCheck = rewriter.create<emitc::CmpOp>(
//         op.getLoc(), rewriter.getI1Type(), emitc::CmpPredicate::lt, rhs, width);

//     // Any concrete value is a valid refinement of poison.
//     Value poison = rewriter.create<emitc::ConstantOp>(
//         op.getLoc(), arithmeticType,
//         (isa<IntegerType>(arithmeticType)
//              ? rewriter.getIntegerAttr(arithmeticType, 0)
//              : rewriter.getIndexAttr(0)));

//     emitc::ExpressionOp ternary = rewriter.create<emitc::ExpressionOp>(
//         op.getLoc(), arithmeticType, /*do_not_inline=*/false);
//     Block &bodyBlock = ternary.getBodyRegion().emplaceBlock();
//     auto currentPoint = rewriter.getInsertionPoint();
//     rewriter.setInsertionPointToStart(&bodyBlock);
//     Value arithmeticResult =
//         rewriter.create<EmitCOp>(op.getLoc(), arithmeticType, lhs, rhs);
//     Value resultOrPoison = rewriter.create<emitc::ConditionalOp>(
//         op.getLoc(), arithmeticType, excessCheck, arithmeticResult, poison);
//     rewriter.create<emitc::YieldOp>(op.getLoc(), resultOrPoison);
//     rewriter.setInsertionPoint(op->getBlock(), currentPoint);

//     Value result = adaptValueType(ternary, rewriter, type);

//     rewriter.replaceOp(op, result);
//     return success();
//   }
// };

// template <typename ArithOp, typename EmitCOp>
// class SignedShiftOpConversion final
//     : public ShiftOpConversion<ArithOp, EmitCOp, false> {
//   using ShiftOpConversion<ArithOp, EmitCOp, false>::ShiftOpConversion;
// };

// template <typename ArithOp, typename EmitCOp>
// class UnsignedShiftOpConversion final
//     : public ShiftOpConversion<ArithOp, EmitCOp, true> {
//   using ShiftOpConversion<ArithOp, EmitCOp, true>::ShiftOpConversion;
// };

// class SelectOpConversion : public OpConversionPattern<arith::SelectOp> {
// public:
//   using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(arith::SelectOp selectOp, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type dstType = getTypeConverter()->convertType(selectOp.getType());
//     if (!dstType)
//       return rewriter.notifyMatchFailure(selectOp, "type conversion failed");

//     if (!adaptor.getCondition().getType().isInteger(1))
//       return rewriter.notifyMatchFailure(
//           selectOp,
//           "can only be converted if condition is a scalar of type i1");

//     rewriter.replaceOpWithNewOp<emitc::ConditionalOp>(selectOp, dstType,
//                                                       adaptor.getOperands());

//     return success();
//   }
// };

// // Floating-point to integer conversions.
// template <typename CastOp>
// class FtoICastOpConversion : public OpConversionPattern<CastOp> {
// public:
//   FtoICastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
//       : OpConversionPattern<CastOp>(typeConverter, context) {}

//   LogicalResult
//   matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {

//     Type operandType = adaptor.getIn().getType();
//     if (!emitc::isSupportedFloatType(operandType))
//       return rewriter.notifyMatchFailure(castOp,
//                                          "unsupported cast source type");

//     Type dstType = this->getTypeConverter()->convertType(castOp.getType());
//     if (!dstType)
//       return rewriter.notifyMatchFailure(castOp, "type conversion failed");

//     // Float-to-i1 casts are not supported: any value with 0 < value < 1 must be
//     // truncated to 0, whereas a boolean conversion would return true.
//     if (!emitc::isSupportedIntegerType(dstType) || dstType.isInteger(1))
//       return rewriter.notifyMatchFailure(castOp,
//                                          "unsupported cast destination type");

//     // Convert to unsigned if it's the "ui" variant
//     // Signless is interpreted as signed, so no need to cast for "si"
//     Type actualResultType = dstType;
//     if (isa<arith::FPToUIOp>(castOp)) {
//       actualResultType =
//           rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
//                                   /*isSigned=*/false);
//     }

//     Value result = rewriter.create<emitc::CastOp>(
//         castOp.getLoc(), actualResultType, adaptor.getOperands());

//     if (isa<arith::FPToUIOp>(castOp)) {
//       result = rewriter.create<emitc::CastOp>(castOp.getLoc(), dstType, result);
//     }
//     rewriter.replaceOp(castOp, result);

//     return success();
//   }
// };

// // Integer to floating-point conversions.
// template <typename CastOp>
// class ItoFCastOpConversion : public OpConversionPattern<CastOp> {
// public:
//   ItoFCastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
//       : OpConversionPattern<CastOp>(typeConverter, context) {}

//   LogicalResult
//   matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const override {
//     // Vectors in particular are not supported
//     Type operandType = adaptor.getIn().getType();
//     if (!emitc::isSupportedIntegerType(operandType))
//       return rewriter.notifyMatchFailure(castOp,
//                                          "unsupported cast source type");

//     Type dstType = this->getTypeConverter()->convertType(castOp.getType());
//     if (!dstType)
//       return rewriter.notifyMatchFailure(castOp, "type conversion failed");

//     if (!emitc::isSupportedFloatType(dstType))
//       return rewriter.notifyMatchFailure(castOp,
//                                          "unsupported cast destination type");

//     // Convert to unsigned if it's the "ui" variant
//     // Signless is interpreted as signed, so no need to cast for "si"
//     Type actualOperandType = operandType;
//     if (isa<arith::UIToFPOp>(castOp)) {
//       actualOperandType =
//           rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
//                                   /*isSigned=*/false);
//     }
//     Value fpCastOperand = adaptor.getIn();
//     if (actualOperandType != operandType) {
//       fpCastOperand = rewriter.template create<emitc::CastOp>(
//           castOp.getLoc(), actualOperandType, fpCastOperand);
//     }
//     rewriter.replaceOpWithNewOp<emitc::CastOp>(castOp, dstType, fpCastOperand);

//     return success();
//   }
// };

// } // namespace

// //===----------------------------------------------------------------------===//
// // Pattern population
// //===----------------------------------------------------------------------===//

// void mlir::populateArithToEmitCPatterns(TypeConverter &typeConverter,
//                                         RewritePatternSet &patterns) {
//   MLIRContext *ctx = patterns.getContext();

//   mlir::populateEmitCSizeTTypeConversions(typeConverter);

//   // clang-format off
//   patterns.add<
//     ArithConstantOpConversionPattern,
//     ArithOpConversion<arith::AddFOp, emitc::AddOp>,
//     ArithOpConversion<arith::DivFOp, emitc::DivOp>,
//     ArithOpConversion<arith::DivSIOp, emitc::DivOp>,
//     ArithOpConversion<arith::MulFOp, emitc::MulOp>,
//     ArithOpConversion<arith::RemSIOp, emitc::RemOp>,
//     ArithOpConversion<arith::SubFOp, emitc::SubOp>,
//     IntegerOpConversion<arith::AddIOp, emitc::AddOp>,
//     IntegerOpConversion<arith::MulIOp, emitc::MulOp>,
//     IntegerOpConversion<arith::SubIOp, emitc::SubOp>,
//     BitwiseOpConversion<arith::AndIOp, emitc::BitwiseAndOp>,
//     BitwiseOpConversion<arith::OrIOp, emitc::BitwiseOrOp>,
//     BitwiseOpConversion<arith::XOrIOp, emitc::BitwiseXorOp>,
//     UnsignedShiftOpConversion<arith::ShLIOp, emitc::BitwiseLeftShiftOp>,
//     SignedShiftOpConversion<arith::ShRSIOp, emitc::BitwiseRightShiftOp>,
//     UnsignedShiftOpConversion<arith::ShRUIOp, emitc::BitwiseRightShiftOp>,
//     CmpFOpConversion,
//     CmpIOpConversion,
//     NegFOpConversion,
//     SelectOpConversion,
//     // Truncation is guaranteed for unsigned types.
//     UnsignedCastConversion<arith::TruncIOp>,
//     SignedCastConversion<arith::ExtSIOp>,
//     UnsignedCastConversion<arith::ExtUIOp>,
//     SignedCastConversion<arith::IndexCastOp>,
//     UnsignedCastConversion<arith::IndexCastUIOp>,
//     ItoFCastOpConversion<arith::SIToFPOp>,
//     ItoFCastOpConversion<arith::UIToFPOp>,
//     FtoICastOpConversion<arith::FPToSIOp>,
//     FtoICastOpConversion<arith::FPToUIOp>
//   >(typeConverter, ctx);
//   // clang-format on
// }



//--------------------------------------------------------分割线--------------------------------------------------------------------------------


//===- ArithToEmitC.cpp - Arith to EmitC Patterns ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the Arith dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

// #include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"

// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/EmitC/IR/EmitC.h"
// #include "mlir/Tools/PDLL/AST/Types.h"
// #include "mlir/Transforms/DialectConversion.h"



#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
class ArithConstantOpConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp arithConst,
                  arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        arithConst, arithConst.getType(), adaptor.getValue());
    return success();
  }
};

class CmpIOpConversion : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  bool needsUnsignedCmp(arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge:
      return false;
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::uge:
      return true;
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  emitc::CmpPredicate toEmitCPred(arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::eq:
      return emitc::CmpPredicate::eq;
    case arith::CmpIPredicate::ne:
      return emitc::CmpPredicate::ne;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return emitc::CmpPredicate::lt;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return emitc::CmpPredicate::le;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return emitc::CmpPredicate::gt;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return emitc::CmpPredicate::ge;
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = adaptor.getLhs().getType();
    if (!isa_and_nonnull<IntegerType, IndexType>(type)) {
      return rewriter.notifyMatchFailure(op, "expected integer or index type");
    }

    bool needsUnsigned = needsUnsignedCmp(op.getPredicate());
    emitc::CmpPredicate pred = toEmitCPred(op.getPredicate());
    Type arithmeticType = type;
    if (type.isUnsignedInteger() != needsUnsigned) {
      arithmeticType = rewriter.getIntegerType(type.getIntOrFloatBitWidth(),
                                               /*isSigned=*/!needsUnsigned);
    }
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (arithmeticType != type) {
      lhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    lhs);
      rhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    rhs);
    }
    rewriter.replaceOpWithNewOp<emitc::CmpOp>(op, op.getType(), pred, lhs, rhs);
    return success();
  }
};

template <typename ArithOp, bool castToUnsigned>
class CastConversion : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type opReturnType = this->getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(opReturnType))
      return rewriter.notifyMatchFailure(op, "expected integer result type");

    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "CastConversion only supports unary ops");
    }

    Type operandType = adaptor.getIn().getType();
    if (!isa_and_nonnull<IntegerType>(operandType))
      return rewriter.notifyMatchFailure(op, "expected integer operand type");

    // Signed (sign-extending) casts from i1 are not supported.
    if (operandType.isInteger(1) && !castToUnsigned)
      return rewriter.notifyMatchFailure(op,
                                         "operation not supported on i1 type");

    // to-i1 conversions: arith semantics want truncation, whereas (bool)(v) is
    // equivalent to (v != 0). Implementing as (bool)(v & 0x01) gives
    // truncation.
    if (opReturnType.isInteger(1)) {
      auto constOne = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), operandType, rewriter.getIntegerAttr(operandType, 1));
      auto oneAndOperand = rewriter.create<emitc::BitwiseAndOp>(
          op.getLoc(), operandType, adaptor.getIn(), constOne);
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, opReturnType,
                                                 oneAndOperand);
      return success();
    }

    bool isTruncation = operandType.getIntOrFloatBitWidth() >
                        opReturnType.getIntOrFloatBitWidth();
    bool doUnsigned = castToUnsigned || isTruncation;

    Type castType = opReturnType;
    // If the op is a ui variant and the type wanted as
    // return type isn't unsigned, we need to issue an unsigned type to do
    // the conversion.
    if (castType.isUnsignedInteger() != doUnsigned) {
      castType = rewriter.getIntegerType(opReturnType.getIntOrFloatBitWidth(),
                                         /*isSigned=*/!doUnsigned);
    }

    Value actualOp = adaptor.getIn();
    // Adapt the signedness of the operand if necessary
    if (operandType.isUnsignedInteger() != doUnsigned) {
      Type correctSignednessType =
          rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/!doUnsigned);
      actualOp = rewriter.template create<emitc::CastOp>(
          op.getLoc(), correctSignednessType, actualOp);
    }

    auto result = rewriter.template create<emitc::CastOp>(op.getLoc(), castType,
                                                          actualOp);

    // Cast to the expected output type
    if (castType != opReturnType) {
      result = rewriter.template create<emitc::CastOp>(op.getLoc(),
                                                       opReturnType, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ArithOp>
class UnsignedCastConversion : public CastConversion<ArithOp, true> {
  using CastConversion<ArithOp, true>::CastConversion;
};

template <typename ArithOp>
class SignedCastConversion : public CastConversion<ArithOp, false> {
  using CastConversion<ArithOp, false>::CastConversion;
};

template <typename ArithOp, typename EmitCOp>
class ArithOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp arithOp, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.template replaceOpWithNewOp<EmitCOp>(arithOp, arithOp.getType(),
                                                  adaptor.getOperands());

    return success();
  }
};

template <typename ArithOp, typename EmitCOp>
class IntegerOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType, IndexType>(type)) {
      return rewriter.notifyMatchFailure(op, "expected integer type");
    }

    if (type.isInteger(1)) {
      // arith expects wrap-around arithmethic, which doesn't happen on `bool`.
      return rewriter.notifyMatchFailure(op, "i1 type is not implemented");
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Type arithmeticType = type;
    if ((type.isSignlessInteger() || type.isSignedInteger()) &&
        !bitEnumContainsAll(op.getOverflowFlags(),
                            arith::IntegerOverflowFlags::nsw)) {
      // If the C type is signed and the op doesn't guarantee "No Signed Wrap",
      // we compute in unsigned integers to avoid UB.
      arithmeticType = rewriter.getIntegerType(type.getIntOrFloatBitWidth(),
                                               /*isSigned=*/false);
    }
    if (arithmeticType != type) {
      lhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    lhs);
      rhs = rewriter.template create<emitc::CastOp>(op.getLoc(), arithmeticType,
                                                    rhs);
    }

    Value result = rewriter.template create<EmitCOp>(op.getLoc(),
                                                     arithmeticType, lhs, rhs);

    if (arithmeticType != type) {
      result =
          rewriter.template create<emitc::CastOp>(op.getLoc(), type, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp selectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type dstType = getTypeConverter()->convertType(selectOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(selectOp, "type conversion failed");

    if (!adaptor.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          selectOp,
          "can only be converted if condition is a scalar of type i1");

    rewriter.replaceOpWithNewOp<emitc::ConditionalOp>(selectOp, dstType,
                                                      adaptor.getOperands());

    return success();
  }
};

// Floating-point to integer conversions.
template <typename CastOp>
class FtoICastOpConversion : public OpConversionPattern<CastOp> {
public:
  FtoICastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type operandType = adaptor.getIn().getType();
    if (!emitc::isSupportedFloatType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    // Float-to-i1 casts are not supported: any value with 0 < value < 1 must be
    // truncated to 0, whereas a boolean conversion would return true.
    if (!emitc::isSupportedIntegerType(dstType) || dstType.isInteger(1))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    // Convert to unsigned if it's the "ui" variant
    // Signless is interpreted as signed, so no need to cast for "si"
    Type actualResultType = dstType;
    if (isa<arith::FPToUIOp>(castOp)) {
      actualResultType =
          rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/false);
    }

    Value result = rewriter.create<emitc::CastOp>(
        castOp.getLoc(), actualResultType, adaptor.getOperands());

    if (isa<arith::FPToUIOp>(castOp)) {
      result = rewriter.create<emitc::CastOp>(castOp.getLoc(), dstType, result);
    }
    rewriter.replaceOp(castOp, result);

    return success();
  }
};

// Integer to floating-point conversions.
template <typename CastOp>
class ItoFCastOpConversion : public OpConversionPattern<CastOp> {
public:
  ItoFCastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Vectors in particular are not supported
    Type operandType = adaptor.getIn().getType();
    if (!emitc::isSupportedIntegerType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    if (!emitc::isSupportedFloatType(dstType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    // Convert to unsigned if it's the "ui" variant
    // Signless is interpreted as signed, so no need to cast for "si"
    Type actualOperandType = operandType;
    if (isa<arith::UIToFPOp>(castOp)) {
      actualOperandType =
          rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/false);
    }
    Value fpCastOperand = adaptor.getIn();
    if (actualOperandType != operandType) {
      fpCastOperand = rewriter.template create<emitc::CastOp>(
          castOp.getLoc(), actualOperandType, fpCastOperand);
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(castOp, dstType, fpCastOperand);

    return success();
  }
};

class NegFOpConversion : public OpConversionPattern<arith::NegFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto adaptedOp = adaptor.getOperand();
    auto adaptedOpType = adaptedOp.getType();

    if (isa<TensorType>(adaptedOpType) || isa<VectorType>(adaptedOpType)) {
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "negf currently only supports scalar types, not vectors or tensors");
    }

    if (!emitc::isSupportedFloatType(adaptedOpType)) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "floating-point type is not supported by EmitC");
    }

    rewriter.replaceOpWithNewOp<emitc::UnaryMinusOp>(op, adaptedOpType,
                                                     adaptedOp);
    return success();
  }
};


class CmpFOpConversion : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!isa<FloatType>(adaptor.getRhs().getType())) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cmpf currently only supported on "
                                         "floats, not tensors/vectors thereof");
    }

    bool unordered = false;
    emitc::CmpPredicate predicate;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse: {
      auto constant = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(),
          rewriter.getBoolAttr(/*value=*/false));
      rewriter.replaceOp(op, constant);
      return success();
    }
    case arith::CmpFPredicate::OEQ:
      unordered = false;
      predicate = emitc::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::OGT:
      unordered = false;
      predicate = emitc::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::OGE:
      unordered = false;
      predicate = emitc::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::OLT:
      unordered = false;
      predicate = emitc::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::OLE:
      unordered = false;
      predicate = emitc::CmpPredicate::le;
      break;
    case arith::CmpFPredicate::ONE:
      unordered = false;
      predicate = emitc::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::ORD: {
      // ordered, i.e. none of the operands is NaN
      auto cmp = createCheckIsOrdered(rewriter, op.getLoc(), adaptor.getLhs(),
                                      adaptor.getRhs());
      rewriter.replaceOp(op, cmp);
      return success();
    }
    case arith::CmpFPredicate::UEQ:
      unordered = true;
      predicate = emitc::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::UGT:
      unordered = true;
      predicate = emitc::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::UGE:
      unordered = true;
      predicate = emitc::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::ULT:
      unordered = true;
      predicate = emitc::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::ULE:
      unordered = true;
      predicate = emitc::CmpPredicate::le;
      break;
    case arith::CmpFPredicate::UNE:
      unordered = true;
      predicate = emitc::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::UNO: {
      // unordered, i.e. either operand is nan
      auto cmp = createCheckIsUnordered(rewriter, op.getLoc(), adaptor.getLhs(),
                                        adaptor.getRhs());
      rewriter.replaceOp(op, cmp);
      return success();
    }
    case arith::CmpFPredicate::AlwaysTrue: {
      auto constant = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(),
          rewriter.getBoolAttr(/*value=*/true));
      rewriter.replaceOp(op, constant);
      return success();
    }
    }

    // Compare the values naively
    auto cmpResult =
        rewriter.create<emitc::CmpOp>(op.getLoc(), op.getType(), predicate,
                                      adaptor.getLhs(), adaptor.getRhs());

    // Adjust the results for unordered/ordered semantics
    if (unordered) {
      auto isUnordered = createCheckIsUnordered(
          rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<emitc::LogicalOrOp>(op, op.getType(),
                                                      isUnordered, cmpResult);
      return success();
    }

    auto isOrdered = createCheckIsOrdered(rewriter, op.getLoc(),
                                          adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<emitc::LogicalAndOp>(op, op.getType(),
                                                     isOrdered, cmpResult);
    return success();
  }

private:
  /// Return a value that is true if \p operand is NaN.
  Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
              Value operand) const {
    // A value is NaN exactly when it compares unequal to itself.
    return rewriter.create<emitc::CmpOp>(
        loc, rewriter.getI1Type(), emitc::CmpPredicate::ne, operand, operand);
  }

  /// Return a value that is true if \p operand is not NaN.
  Value isNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                 Value operand) const {
    // A value is not NaN exactly when it compares equal to itself.
    return rewriter.create<emitc::CmpOp>(
        loc, rewriter.getI1Type(), emitc::CmpPredicate::eq, operand, operand);
  }

  /// Return a value that is true if the operands \p first and \p second are
  /// unordered (i.e., at least one of them is NaN).
  Value createCheckIsUnordered(ConversionPatternRewriter &rewriter,
                               Location loc, Value first, Value second) const {
    auto firstIsNaN = isNaN(rewriter, loc, first);
    auto secondIsNaN = isNaN(rewriter, loc, second);
    return rewriter.create<emitc::LogicalOrOp>(loc, rewriter.getI1Type(),
                                               firstIsNaN, secondIsNaN);
  }

  /// Return a value that is true if the operands \p first and \p second are
  /// both ordered (i.e., none one of them is NaN).
  Value createCheckIsOrdered(ConversionPatternRewriter &rewriter, Location loc,
                             Value first, Value second) const {
    auto firstIsNotNaN = isNotNaN(rewriter, loc, first);
    auto secondIsNotNaN = isNotNaN(rewriter, loc, second);
    return rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                firstIsNotNaN, secondIsNotNaN);
  }
};

class IndexCastOpConversion : public OpConversionPattern<arith::IndexCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    Type sourceType = adaptor.getIn().getType();
    Type targetType = getTypeConverter()->convertType(op.getType());
    
    if (!targetType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    // index_cast 在 EmitC 中通过简单的 cast 操作实现
    // index 类型会被 TypeConverter 转换为具体的整数类型
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, targetType, adaptor.getIn());
    
    return success();
  }
};

// ───────────────────────────────────────────────────────────────────────
// 移位操作转换
//
// arith.shrui / arith.shrsi / arith.shli 均映射到对应的 emitc bitwise op。
// shrui 需要将操作数先 cast 为无符号类型，确保生成 C 的逻辑右移（>>）
// 而非算术右移，语义与 arith.shrui 一致。
// ───────────────────────────────────────────────────────────────────────

// arith.shrui (i64, i64) -> i64  =>  emitc.bitwise_right_shift (unsigned)
class ShrUIOpConversion : public OpConversionPattern<arith::ShRUIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(type))
      return rewriter.notifyMatchFailure(op, "expected integer type");
    if (type.isInteger(1))
      return rewriter.notifyMatchFailure(op, "i1 shift not supported");

    // shrui 必须在无符号类型上操作，否则 C 的 >> 是实现定义行为
    Type unsignedType = rewriter.getIntegerType(
        type.getIntOrFloatBitWidth(), /*isSigned=*/false);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // 若当前类型不是无符号，先 cast
    if (type != unsignedType) {
      lhs = rewriter.create<emitc::CastOp>(op.getLoc(), unsignedType, lhs);
      rhs = rewriter.create<emitc::CastOp>(op.getLoc(), unsignedType, rhs);
    }

    Value result = rewriter.create<emitc::BitwiseRightShiftOp>(
        op.getLoc(), unsignedType, lhs, rhs);

    // 结果 cast 回原类型（signless integer）
    if (unsignedType != type)
      result = rewriter.create<emitc::CastOp>(op.getLoc(), type, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// arith.shrsi (signed arithmetic right shift)
class ShrSIOpConversion : public OpConversionPattern<arith::ShRSIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(type))
      return rewriter.notifyMatchFailure(op, "expected integer type");
    if (type.isInteger(1))
      return rewriter.notifyMatchFailure(op, "i1 shift not supported");

    // shrsi 保持有符号类型，C 的 >> 对有符号整数是算术右移（实现定义，
    // 但所有目标平台均如此），与 arith.shrsi 语义一致
    Type signedType = rewriter.getIntegerType(
        type.getIntOrFloatBitWidth(), /*isSigned=*/true);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (type != signedType) {
      lhs = rewriter.create<emitc::CastOp>(op.getLoc(), signedType, lhs);
      rhs = rewriter.create<emitc::CastOp>(op.getLoc(), signedType, rhs);
    }

    Value result = rewriter.create<emitc::BitwiseRightShiftOp>(
        op.getLoc(), signedType, lhs, rhs);

    if (signedType != type)
      result = rewriter.create<emitc::CastOp>(op.getLoc(), type, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// arith.shli (left shift)
class ShlIOpConversion : public OpConversionPattern<arith::ShLIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(type))
      return rewriter.notifyMatchFailure(op, "expected integer type");
    if (type.isInteger(1))
      return rewriter.notifyMatchFailure(op, "i1 shift not supported");

    // 左移在无符号类型上操作，避免有符号溢出 UB
    Type unsignedType = rewriter.getIntegerType(
        type.getIntOrFloatBitWidth(), /*isSigned=*/false);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (type != unsignedType) {
      lhs = rewriter.create<emitc::CastOp>(op.getLoc(), unsignedType, lhs);
      rhs = rewriter.create<emitc::CastOp>(op.getLoc(), unsignedType, rhs);
    }

    Value result = rewriter.create<emitc::BitwiseLeftShiftOp>(
        op.getLoc(), unsignedType, lhs, rhs);

    if (unsignedType != type)
      result = rewriter.create<emitc::CastOp>(op.getLoc(), type, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// ───────────────────────────────────────────────────────────────────────
// 按位逻辑操作转换
// arith.andi/ori/xori → emitc.bitwise_and/or/xor
// 注意：这里使用无符号类型进行操作，与 C 的位运算语义一致
// ───────────────────────────────────────────────────────────────────────
template <typename ArithOp, typename EmitCOp>
class BitwiseOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(type))
      return rewriter.notifyMatchFailure(op, "expected integer type");
    if (type.isInteger(1))
      return rewriter.notifyMatchFailure(op, "i1 not supported");

    // 按位操作在无符号类型上进行，避免实现定义行为
    Type unsignedType = rewriter.getIntegerType(
        type.getIntOrFloatBitWidth(), /*isSigned=*/false);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (type != unsignedType) {
      lhs = rewriter.template create<emitc::CastOp>(op.getLoc(), unsignedType, lhs);
      rhs = rewriter.template create<emitc::CastOp>(op.getLoc(), unsignedType, rhs);
    }

    Value result = rewriter.template create<EmitCOp>(
        op.getLoc(), unsignedType, lhs, rhs);

    if (unsignedType != type)
      result = rewriter.template create<emitc::CastOp>(op.getLoc(), type, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// ───────────────────────────────────────────────────────────────────────
// arith.bitcast 转换
//
// arith.bitcast 的语义：重新解释操作数的位模式为目标类型，
// 等价于 C 中的 memcpy-based type punning（C99/C11 合法写法）或
// C++20 的 std::bit_cast。
//
// 生成形式：emitc.call_opaque "bit_cast_i32_f32"(%val) 不够通用，
// 改用 emitc.verbatim 或 emitc.call_opaque + 宏，最简洁的方案是
// 利用 GCC/Clang 均支持的 __builtin_bit_cast（C++20 前的扩展）：
//
//   __builtin_bit_cast(dst_type, src_value)
//
// 在 emitc 中通过 emitc.call_opaque 表达：
//   emitc.call_opaque "__builtin_bit_cast"(%src) {args=[#emitc.opaque<"float">]}
//
// 注意：仅支持 i32↔f32、i64↔f64 等等宽整数↔浮点的互转。
// ───────────────────────────────────────────────────────────────────────
class BitcastOpConversion : public OpConversionPattern<arith::BitcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type srcType = adaptor.getIn().getType();
    Type dstType = getTypeConverter()->convertType(op.getType());

    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    // 只支持标量整数↔浮点的等宽互转
    auto isIntOrFloat = [](Type t) {
      return isa<IntegerType>(t) || isa<FloatType>(t);
    };
    if (!isIntOrFloat(srcType) || !isIntOrFloat(dstType))
      return rewriter.notifyMatchFailure(
          op, "bitcast only supported between scalar int and float types");

    // 获取目标类型对应的 C 类型名称字符串
    // emitc 的 OpaqueType 可以携带任意 C 类型字符串
    auto getCTypeName = [&](Type t) -> std::optional<std::string> {
      if (t.isF32())       return "float";
      if (t.isF64())       return "double";
      if (t.isInteger(32)) return "int32_t";
      if (t.isInteger(64)) return "int64_t";
      if (t.isInteger(16)) return "int16_t";
      if (t.isInteger(8))  return "int8_t";
      return std::nullopt;
    };

    auto dstCName = getCTypeName(dstType);
    if (!dstCName)
      return rewriter.notifyMatchFailure(op, "unsupported bitcast target type");

    // 生成：__builtin_bit_cast(dst_type, src)
    // emitc.call_opaque 的 args 属性携带第一个参数（类型名），
    // 操作数携带第二个参数（值）
    //
    // 等价 C 代码：__builtin_bit_cast(float, some_i32_value)
    auto dstOpaqueType = emitc::OpaqueType::get(rewriter.getContext(),
                                                 *dstCName);
    // 用 emitc.call_opaque 生成函数调用形式：
    //   (float)(__builtin_bit_cast(float, x))
    // 注意：__builtin_bit_cast 是编译器内建，第一个参数是类型，第二个是值。
    // emitc.call_opaque 不能直接表达「类型参数」，改用
    // emitc.opaque 表达式包装：
    //
    //   emitc.expression : 生成 C 表达式
    //
    // 最简单且 emitc 完全支持的方法：
    // 生成一个带类型注释的 call_opaque，callee 字符串直接包含类型部分。
    //
    // 实际上 __builtin_bit_cast 在 emitc 中最干净的表达方式是
    // 利用 args 数组携带 #emitc.opaque<"float"> 作为第一个（类型）参数：
    //
    //   emitc.call_opaque "__builtin_bit_cast"(%val)
    //       {args = [#emitc.opaque<"float">, 0 : index]} : ...
    //
    // 这样生成的 C 代码正好是：__builtin_bit_cast(float, val)

    SmallVector<Attribute> args;
    // 第一个参数：目标类型名（作为 opaque 字符串传入）
    args.push_back(emitc::OpaqueAttr::get(rewriter.getContext(), *dstCName));
    // 第二个参数：操作数索引 0（即 adaptor.getIn()）
    args.push_back(rewriter.getIndexAttr(0));

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op,
        /*resultTypes=*/TypeRange{dstType},
        /*callee=*/"__builtin_bit_cast",
        /*args=*/rewriter.getArrayAttr(args),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{adaptor.getIn()});

    return success();
  }
};

// ───────────────────────────────────────────────────────────────────────
// arith.maxnumf / arith.minnumf / arith.maximumf / arith.minimumf 转换
//
// maxnumf/minnumf：IEEE 754 maxNum/minNum，NaN 时返回另一个操作数
//   → fmaxf(f32) / fmax(f64) / fminf(f32) / fmin(f64)
//
// maximumf/minimumf：如果任一操作数为 NaN 则返回 NaN
//   → 需要手动实现，这里用 emitc.call_opaque 调用辅助宏或内联条件表达式
//   → 简化处理：目标平台无 NaN 传播需求时，同样映射到 fmaxf/fminf
// ───────────────────────────────────────────────────────────────────────

template <typename ArithOp>
class FloatMinMaxOpConversion : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  // 根据 op 类型和元素位宽选择对应的 C 函数名
  static StringRef getCFuncName(bool isMax, unsigned bitWidth) {
    if (isMax)
      return bitWidth == 32 ? "fmaxf" : "fmax";
    else
      return bitWidth == 32 ? "fminf" : "fmin";
  }

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type srcType = adaptor.getLhs().getType();
    if (!isa<FloatType>(srcType))
      return rewriter.notifyMatchFailure(op, "expected float type");

    Type dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    unsigned bitWidth = srcType.getIntOrFloatBitWidth();
    if (bitWidth != 32 && bitWidth != 64)
      return rewriter.notifyMatchFailure(op,
          "only f32 and f64 are supported for float min/max");

    // isMax: maxnumf/maximumf → true, minnumf/minimumf → false
    constexpr bool isMax =
        std::is_same_v<ArithOp, arith::MaxNumFOp> ||
        std::is_same_v<ArithOp, arith::MaximumFOp>;

    StringRef funcName = getCFuncName(isMax, bitWidth);

    // 生成：fmaxf(lhs, rhs) 或 fminf(lhs, rhs)
    // args 中只放操作数索引，不放额外的类型参数
    SmallVector<Attribute> args;
    args.push_back(rewriter.getIndexAttr(0));  // lhs
    args.push_back(rewriter.getIndexAttr(1));  // rhs

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op,
        /*resultTypes=*/TypeRange{dstType},
        /*callee=*/funcName,
        /*args=*/rewriter.getArrayAttr(args),
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()});

    return success();
  }
};

// 整数 max/min：arith.maxsi / arith.minsi / arith.maxui / arith.minui
// 映射到三目运算符，通过 emitc.conditional 表达
template <typename ArithOp, bool isMax, bool isUnsigned>
class IntMinMaxOpConversion : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(type))
      return rewriter.notifyMatchFailure(op, "expected integer type");

    // 根据有符号/无符号选择比较类型
    Type cmpType = isUnsigned
        ? rewriter.getIntegerType(type.getIntOrFloatBitWidth(), /*isSigned=*/false)
        : rewriter.getIntegerType(type.getIntOrFloatBitWidth(), /*isSigned=*/true);

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // 如需调整符号性，先 cast
    if (type != cmpType) {
      lhs = rewriter.template create<emitc::CastOp>(op.getLoc(), cmpType, lhs);
      rhs = rewriter.template create<emitc::CastOp>(op.getLoc(), cmpType, rhs);
    }

    // 生成比较：lhs > rhs (max) 或 lhs < rhs (min)
    emitc::CmpPredicate pred = isMax ? emitc::CmpPredicate::gt
                                     : emitc::CmpPredicate::lt;
    Value cond = rewriter.create<emitc::CmpOp>(
        op.getLoc(), rewriter.getI1Type(), pred, lhs, rhs);

    // 生成三目：cond ? lhs : rhs
    Value result = rewriter.create<emitc::ConditionalOp>(
        op.getLoc(), cmpType, cond, lhs, rhs);

    // cast 回原类型
    if (type != cmpType)
      result = rewriter.create<emitc::CastOp>(op.getLoc(), type, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} 
// namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateArithToEmitCPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // clang-format off
  patterns.add<
    ArithConstantOpConversionPattern,
    ArithOpConversion<arith::AddFOp, emitc::AddOp>,
    ArithOpConversion<arith::DivFOp, emitc::DivOp>,
    ArithOpConversion<arith::DivSIOp, emitc::DivOp>,
    ArithOpConversion<arith::MulFOp, emitc::MulOp>,
    ArithOpConversion<arith::RemSIOp, emitc::RemOp>,
    ArithOpConversion<arith::SubFOp, emitc::SubOp>,
    IntegerOpConversion<arith::AddIOp, emitc::AddOp>,
    IntegerOpConversion<arith::MulIOp, emitc::MulOp>,
    IntegerOpConversion<arith::SubIOp, emitc::SubOp>,
    CmpIOpConversion,
    CmpFOpConversion,
    NegFOpConversion,
    SelectOpConversion,
    IndexCastOpConversion,
    // Truncation is guaranteed for unsigned types.
    UnsignedCastConversion<arith::TruncIOp>,
    SignedCastConversion<arith::ExtSIOp>,
    UnsignedCastConversion<arith::ExtUIOp>,
    ItoFCastOpConversion<arith::SIToFPOp>,
    ItoFCastOpConversion<arith::UIToFPOp>,
    FtoICastOpConversion<arith::FPToSIOp>,
    FtoICastOpConversion<arith::FPToUIOp>,
    // 移位操作
    ShrUIOpConversion,
    ShrSIOpConversion,
    ShlIOpConversion,
    // 按位逻辑操作
    BitwiseOpConversion<arith::AndIOp, emitc::BitwiseAndOp>,
    BitwiseOpConversion<arith::OrIOp,  emitc::BitwiseOrOp>,
    BitwiseOpConversion<arith::XOrIOp, emitc::BitwiseXorOp>,
    BitcastOpConversion,
    // 浮点 min/max
    FloatMinMaxOpConversion<arith::MaxNumFOp>,
    FloatMinMaxOpConversion<arith::MinNumFOp>,
    FloatMinMaxOpConversion<arith::MaximumFOp>,
    FloatMinMaxOpConversion<arith::MinimumFOp>,
    // 整数 min/max
    IntMinMaxOpConversion<arith::MaxSIOp, /*isMax=*/true,  /*isUnsigned=*/false>,
    IntMinMaxOpConversion<arith::MinSIOp, /*isMax=*/false, /*isUnsigned=*/false>,
    IntMinMaxOpConversion<arith::MaxUIOp, /*isMax=*/true,  /*isUnsigned=*/true>,
    IntMinMaxOpConversion<arith::MinUIOp, /*isMax=*/false, /*isUnsigned=*/true>
  >(typeConverter, ctx);
  // clang-format on
}
