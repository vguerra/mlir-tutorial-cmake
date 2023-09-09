#include "lib/Dialect/Poly/PolyOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/Poly/PolyOpsDialect.cpp.inc"
#include "mlir/IR/BuiltinAttributes.h"
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

Operation *PolyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, coeffs);
}

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir