#pragma once

#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/syscall.h>
#include <unistd.h>

#include "playground/system.hpp"

namespace playground
{

PJ_FINLINE char* currentTime()
{
    time_t raw_time = time(nullptr);
    struct tm* time_info = localtime(&raw_time);
    static char now_time[64];
    now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S",
                      time_info)] = '\0';

    return now_time;
}

PJ_FINLINE int getPid()
{
    static int pid = getpid();

    return pid;
}

PJ_FINLINE int32_t getTid()
{
    thread_local int32_t tid = syscall(SYS_gettid);

    return tid;
}

#define HGEMM_LOG_TAG "PLAYGROUND"
#define HGEMM_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define HLOG(format, ...)                                                      \
    do {                                                                       \
        fprintf(stderr, "[%s %s %d:%ld %s:%d %s] " format "\n", HGEMM_LOG_TAG, \
                playground::currentTime(), playground::getPid(),               \
                playground::getTid(), HGEMM_LOG_FILE(__FILE__), __LINE__,      \
                __FUNCTION__, ##__VA_ARGS__);                                  \
    } while (0)

}  // namespace playground