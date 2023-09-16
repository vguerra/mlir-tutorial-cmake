#ifndef LIB_DIALECT_POLY_TRAITS_H_
#define LIB_DIALECT_POLY_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir::tutorial::poly {

template <typename ConcreteType>
class Has32BitArguments
    : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      // OK to skip non-integer operand types
      if (!type.isIntOrIndex())
        continue;

      if (!type.isInteger(32)) {
        return op->emitOpError()
               << "requires each numeric operand to be a 32-bit integer";
      }
    }

    return success();
  }
};

} // namespace mlir::tutorial::poly

#endif // LIB_DIALECT_POLY_TRAITS_H_