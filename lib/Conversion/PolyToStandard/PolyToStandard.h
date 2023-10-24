#ifndef LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_H_
#define LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_H_

#include "mlir/Pass/Pass.h"

// Extra includes needed for dependent dialects
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace tutorial {
namespace poly {

#define GEN_PASS_DECL
#include "lib/Conversion/PolyToStandard/PolyToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversion/PolyToStandard/PolyToStandard.h.inc"

} // namespace poly
} // namespace tutorial
} // namespace mlir

#endif // LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_H_