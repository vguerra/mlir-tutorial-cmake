#include "lib/Dialect/Poly/PolyOps.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

// Required after PatternMatch.h
#include "lib/Dialect/Poly/PolyCanonicalize.cpp.inc"

namespace mlir {
namespace tutorial {
namespace poly {

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
    return adaptor.getCoefficients();
}

OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
    return constFoldBinaryOp<IntegerAttr, APInt>(
        adaptor.getOperands(), [&](APInt a, APInt b) {
            return a + b;
        });
}

OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor) {
    return constFoldBinaryOp<IntegerAttr, APInt>(
        adaptor.getOperands(), [&](APInt a, APInt b) {
            return a - b;
        });
}

OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor) {
    auto lhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[0]);
    auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[1]);

    if (!lhs || !rhs)
        return nullptr;

    auto degree = getResult().getType().cast<PolynomialType>().getDegreeBound();
    auto maxIndex = lhs.size() + rhs.size() - 1;

    SmallVector<APInt, 8> result;
    result.reserve(maxIndex);
    for (int i = 0; i < maxIndex; ++i) {
        result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
    }

    int i = 0;
    for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>(); ++lhsIt) {
        int j = 0;
        for (auto rhsIt = rhs.value_begin<APInt>(); rhsIt != rhs.value_end<APInt>(); ++rhsIt) {
            // index is modulo degree because poly's semantics are defined module x^N = 1.
            result[(i + j) % degree] += *rhsIt * (*lhsIt);
            ++j;
        }
        ++i;
    }

    return DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(result.size()),
        IntegerType::get(getContext(), 32)),
        result
    );
}

OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor) {
    // Returns null if the cast failed, which corresponds to a failed fold.
    return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
}

LogicalResult EvalOp::verify() {
    auto pointTy = getPoint().getType();
    bool isSignlessInteger = pointTy.isSignlessInteger(32);
    auto complexPt = llvm::dyn_cast<ComplexType>(pointTy);

    return isSignlessInteger || complexPt
               ? success()
               : emitOpError("argument point must be a 32-bit integer, or a "
                             "complex number");
}

// Rewrites (x^2 - y^2) as (x - y)(x - y) if x^2 and y^2 have not other uses.
struct DifferenceOfSquares : public OpRewritePattern<SubOp> {
  DifferenceOfSquares(MLIRContext *context)
      : OpRewritePattern<SubOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(SubOp op, PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // If either arg has another use, then this rewrite is probably less
    // efficient, because it cannot delete the mul ops.
    if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
        return failure();
    }

    auto rhsMul = rhs.getDefiningOp<MulOp>();
    auto lhsMul = lhs.getDefiningOp<MulOp>();

    if (!rhsMul || !rhsMul) {
        return failure();
    }

    auto x = lhsMul.getLhs();
    auto y = rhsMul.getLhs();

    AddOp newAdd = rewriter.create<AddOp>(op.getLoc(), x, y);
    SubOp newSub = rewriter.create<SubOp>(op.getLoc(), x, y);
    MulOp newMul = rewriter.create<MulOp>(op.getLoc(), newAdd, newSub);

    rewriter.replaceOp(op, {newMul});
    // We don't need to remove the original ops because MLIR already has
    // canonicalization patterns that remove unused ops.

    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.add<DifferenceOfSquares>(context);
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {}

void EvalOp::getCanonicalizationPatterns(RewritePatternSet &results,
 MLIRContext *context) {
    populateWithGenerated(results);
 }

} // namespace poly
} // namespace tutorial
} // namespace mlir