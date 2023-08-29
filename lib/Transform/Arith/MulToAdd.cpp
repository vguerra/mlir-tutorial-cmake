#include "lib/Transform/Arith/MulToAdd.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tutorial {

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {

    Value lhs = op.getOperand(0);

    // canonicalization patterns ensure the constant is on the right, if there
    // is a constant See
    // https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
    Value rhs = op.getOperand(1);
    auto rhsDefinitionOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefinitionOp) {
      return failure();
    }

    int64_t value = rhsDefinitionOp.value();
    bool is_power_of_two = (value & (value - 1)) == 0;

    if (!is_power_of_two) {
      return failure();
    }

    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefinitionOp.getLoc(),
        rewriter.getIntegerAttr(rhs.getType(), value / 2));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefinitionOp);

    return success();
  }
};

//
struct PeelFromMul : public OpRewritePattern<MulIOp> {
  PeelFromMul(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewritter) const override {
    return success();
  }
};

void MulToAddPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<PowerOfTwoExpand>(&getContext());

  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace tutorial
} // namespace mlir