#pragma once

#include <format>
#include <iostream>

#define PLAYGROUND_CHECK(condition)                                           \
    do {                                                                      \
        if (!(condition)) {                                                   \
            ::std::cerr << ::std::format(                                     \
                "[Playground] Check failed at {}:{} for \"{}\"\n", __FILE__,  \
                __LINE__, #condition);                                        \
            ::std::exit(EXIT_FAILURE);                                        \
        }                                                                     \
    } while (0)
