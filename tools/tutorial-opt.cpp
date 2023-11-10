#include "lib/Conversion/PolyToStandard/PolyToStandard.h"
#include "lib/Dialect/Poly/PolyDialect.h"
#include "lib/Transform/Affine/Passes.h"
#include "lib/Transform/Arith/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  // Poly
  manager.addPass(mlir::tutorial::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  registry.insert<mlir::tutorial::poly::PolyDialect>();

  mlir::tutorial::registerAffinePasses();
  mlir::tutorial::registerArithPasses();

  // Dialect conversion passes

  mlir::tutorial::poly::registerPolyToStandardPasses();

  mlir::PassPipelineRegistration<>("poly-to-llvm",
    "Run passes to lower the poly dialect to LLVM",
    polyToLLVMPipelineBuilder
  );

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}