#include "lib/Transform/Affine/AffineUnroll.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();

    mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPassAsPatternRewrite>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry)
    );
}