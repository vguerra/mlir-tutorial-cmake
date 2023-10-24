add_mlir_library(MLIRPolyToStandard
    PolyToStandard.cpp

    ${PROJECT_SOURCE_DIR}/lib/Conversion/PolyToStandard/
    ADDITIONAL_HEADER_DIRS

    DEPENDS
    PolyToStandardPassIncGen

    LINK_LIBS PUBLIC
)

set(LLVM_TARGET_DEFINITIONS PolyToStandard.td)
mlir_tablegen(PolyToStandard.h.inc -gen-pass-decls -name PolyToStandard)
add_dependencies(mlir-headers MLIRPolyOpsIncGen)
add_public_tablegen_target(PolyToStandardPassIncGen)
add_mlir_doc(PolyToStandard PolyToStandard PolyToStandard/ -gen-pass-doc)