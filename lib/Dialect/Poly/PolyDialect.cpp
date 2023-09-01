#include "lib/Dialect/Poly/PolyDialect.h"

#include "lib/Dialect/Poly/PolyTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/Poly/PolyOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly/PolyOpsTypes.cpp.inc"

namespace mlir {
namespace tutorial {
namespace poly {

void PolyDialect::initialize() {
  addTypes<
  #define GET_TYPEDEF_LIST
  #include "lib/Dialect/Poly/PolyOpsTypes.cpp.inc"
  >();
  // This is where we will register types and operations with the dialect
}

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir