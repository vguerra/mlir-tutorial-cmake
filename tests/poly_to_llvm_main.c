#include <stdio.h>

int test_poly_fn(int x);

int main(int argc, char** argv) {
    int i = 1;
    int result = test_poly_fn(i);
    printf("Result: %d\n", result);

    return 0;
}