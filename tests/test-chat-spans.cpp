// fork-only: unit tests of the disk-cache span helpers of common_chat_msg_spans (system_only_context_end,
// first_user_message_pos), moved out of test-chat.cpp; assert_equals is copied from there.

#include "chat.h"
#include "common.h"
#include "log.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>

template <class T> static bool equals(const T & expected, const T & actual) {
    return expected == actual;
}

template <class T> static void assert_equals(const T & expected, const T & actual) {
    if (!equals(expected, actual)) {
        std::ostringstream oss_expected;
        oss_expected << expected;
        std::ostringstream oss_actual;
        oss_actual << actual;
        LOG_ERR("Expected: %s\n", oss_expected.str().c_str());
        LOG_ERR("Actual: %s\n", oss_actual.str().c_str());
        common_log_flush(common_log_main());
        throw std::runtime_error("Test failed");
    }
}

// first_user_message_pos(): the boundary B for the [0,B) shared-context checkpoint: the token
// offset of the FIRST user span (end of the shared leading context), or -1 when there is none.
static void test_first_user_message_pos() {
    LOG_DBG("%s\n", __func__);

    auto span = [](common_chat_role role, size_t pos) {
        common_chat_msg_span s;
        s.role = role;
        s.pos  = pos;
        s.len  = 1;
        return s;
    };

    // empty span set -> no user boundary
    {
        common_chat_msg_spans spans;
        assert_equals<int32_t>(-1, spans.first_user_message_pos());
    }

    // spans present but NO user span (assistant-only) -> -1
    {
        common_chat_msg_spans spans;
        spans.spans = { span(COMMON_CHAT_ROLE_ASSISTANT, 5) };
        assert_equals<int32_t>(-1, spans.first_user_message_pos());
    }

    // leading system context (span[0] not user), then a user span -> the user span's pos
    {
        common_chat_msg_spans spans;
        spans.spans = { span(COMMON_CHAT_ROLE_ASSISTANT, 0), span(COMMON_CHAT_ROLE_USER, 128) };
        assert_equals<int32_t>(128, spans.first_user_message_pos());
    }

    // multi-user conversation -> the FIRST user pos (mirror of last_user_message_pos, opposite end)
    {
        common_chat_msg_spans spans;
        spans.spans = {
            span(COMMON_CHAT_ROLE_USER,      64),
            span(COMMON_CHAT_ROLE_ASSISTANT, 80),
            span(COMMON_CHAT_ROLE_USER,      96),
        };
        assert_equals<int32_t>(64, spans.first_user_message_pos());
        // sanity: last_user_message_pos returns the OTHER end, so B (first) is not the tail boundary.
        assert_equals<int32_t>(96, spans.last_user_message_pos());
    }
}

// system_only_context_end(): the end of the last system span on a request that carries no user span
// (a system-prompt-only pre-cache), -1 otherwise. last_user_message_end(): the end of the last user span.
static void test_system_only_context_end() {
    LOG_DBG("%s\n", __func__);

    auto span = [](common_chat_role role, size_t pos, size_t len) {
        common_chat_msg_span s;
        s.role = role;
        s.pos  = pos;
        s.len  = len;
        return s;
    };

    {
        common_chat_msg_spans spans;
        assert_equals<int32_t>(-1, spans.system_only_context_end());
        assert_equals<int32_t>(-1, spans.last_user_message_end());
    }
    // system then the generation prompt (assistant delimiter): the end of the system span, unaligned
    {
        common_chat_msg_spans spans;
        spans.spans = { span(COMMON_CHAT_ROLE_SYSTEM, 0, 70), span(COMMON_CHAT_ROLE_ASSISTANT, 70, 3) };
        assert_equals<int32_t>(70, spans.system_only_context_end());
        assert_equals<int32_t>(-1, spans.last_user_message_end());
    }
    // two system spans (developer + system): the end of the LAST one
    {
        common_chat_msg_spans spans;
        spans.spans = { span(COMMON_CHAT_ROLE_SYSTEM, 0, 20), span(COMMON_CHAT_ROLE_SYSTEM, 20, 33) };
        assert_equals<int32_t>(53, spans.system_only_context_end());
    }
    // a user span anywhere disables it; last_user_message_end is that span's end
    {
        common_chat_msg_spans spans;
        spans.spans = {
            span(COMMON_CHAT_ROLE_SYSTEM,    0,  64),
            span(COMMON_CHAT_ROLE_USER,      64, 30),
            span(COMMON_CHAT_ROLE_ASSISTANT, 94, 3),
        };
        assert_equals<int32_t>(-1, spans.system_only_context_end());
        assert_equals<int32_t>(94, spans.last_user_message_end());
    }
    // no system span at all (a template whose parser emits no system delimiter): -1
    {
        common_chat_msg_spans spans;
        spans.spans = { span(COMMON_CHAT_ROLE_ASSISTANT, 40, 3) };
        assert_equals<int32_t>(-1, spans.system_only_context_end());
    }
}

int main() {
    test_first_user_message_pos();
    test_system_only_context_end();
    return 0;
}
