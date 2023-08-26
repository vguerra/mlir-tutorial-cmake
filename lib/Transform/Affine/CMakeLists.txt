add_mlir_dialect_library(AffineFullUnroll
    AffineUnroll.h
    AffineUnroll.cpp

    ${PROJECT_SOURCE_DIR}/lib/Transform/Affine/
    ADDITIONAL_HEADER_DIRS
    LINK_LIBS PUBLIC
)