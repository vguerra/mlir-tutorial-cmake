#ifndef LIB_DIALECT_POLY_POLYOPS_H_
#define LIB_DIALECT_POLY_POLYOPS_H_

#include "lib/Dialect/Poly/PolyDialect.h"
#include "lib/Dialect/Poly/PolyTraits.h"
#include "lib/Dialect/Poly/PolyTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Poly/PolyOps.h.inc"

#endif  // LIB_DIALECT_POLY_POLYOPS_H_