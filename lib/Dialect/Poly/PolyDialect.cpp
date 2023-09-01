#include "lib/Dialect/Poly/PolyOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/Poly/PolyOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly/PolyOpsTypes.cpp.inc"
#define  GET_OP_CLASSES
#include "lib/Dialect/Poly/PolyOps.cpp.inc"

namespace mlir {
namespace tutorial {
namespace poly {

void PolyDialect::initialize() {
  addTypes<
  #define GET_TYPEDEF_LIST
  #include "lib/Dialect/Poly/PolyOpsTypes.cpp.inc"
  >();
  addOperations<
  #define GET_OP_LIST
  #include "lib/Dialect/Poly/PolyOps.cpp.inc"
  >();
}

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir