#include "lib/Transform/Affine/AffineUnroll.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Support/LogicalResult.h"

namespace  mlir {
namespace  tutorial {

using  mlir::affine::AffineForOp;
using  mlir::affine::loopUnrollFull;

void AffineFullUnrollPass::runOnOperation() {
    getOperation().walk([&](AffineForOp op) {
        if (failed(loopUnrollFull(op))) {
            op.emitError("unrolling failed");
            signalPassFailure();
        }
    });
    return;
}

} // namespace tutorial
} // namespace mlir