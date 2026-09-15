/// Lightweight test macros — no external dependencies
#include <iostream>
#include <cmath>
#include <cstring>
#include <sstream>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(expr)                                                      \
    do {                                                                        \
        if (!(expr)) {                                                          \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__             \
                      << " — " << #expr << "\n";                              \
            g_tests_failed++;                                                   \
            return;                                                             \
        }                                                                       \
    } while (0)

#define TEST_ASSERT_EQ(a, b)                                                   \
    do {                                                                        \
        auto _a = (a); auto _b = (b);                                          \
        if (_a != _b) {                                                         \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__             \
                      << " — " << #a << " == " << #b                          \
                      << " (got " << _a << " vs " << _b << ")\n";             \
            g_tests_failed++;                                                   \
            return;                                                             \
        }                                                                       \
    } while (0)

#define TEST_ASSERT_NEAR(a, b, eps)                                            \
    do {                                                                        \
        if (std::fabs((a) - (b)) > (eps)) {                                    \
            std::cerr << "  FAIL: " << __FILE__ << ":" << __LINE__             \
                      << " — |" << #a << " - " << #b << "| > " << (eps)       \
                      << " (got " << (a) << " vs " << (b) << ")\n";           \
            g_tests_failed++;                                                   \
            return;                                                             \
        }                                                                       \
    } while (0)

#define RUN_TEST(func)                                                          \
    do {                                                                        \
        std::cout << "  [RUN ] " << #func << "\n";                             \
        func();                                                                 \
        if (g_tests_failed == prev_failed) {                                   \
            std::cout << "  [PASS] " << #func << "\n";                         \
            g_tests_passed++;                                                   \
        }                                                                       \
    } while (0)

#define TEST_MAIN_BEGIN(suite_name)                                             \
    int main() {                                                                \
        std::cout << "\n=== " << suite_name << " ===\n\n";                     \
        int prev_failed = 0;

#define TEST_MAIN_END()                                                         \
        std::cout << "\n  Results: " << g_tests_passed << " passed, "          \
                  << g_tests_failed << " failed\n\n";                          \
        return g_tests_failed > 0 ? 1 : 0;                                    \
    }
