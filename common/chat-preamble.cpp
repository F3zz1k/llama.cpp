// fork-only: common_chat_preamble_end, see chat-preamble.h.

#include "chat-preamble.h"

#include "chat.h"
#include "log.h"

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

int32_t common_chat_preamble_end(const struct common_chat_templates *        tmpls,
                                 const struct common_chat_templates_inputs & inputs,
                                 const std::string &                         prompt) {
    size_t k = 0;
    while (k < inputs.messages.size() &&
           (inputs.messages[k].role == "system" || inputs.messages[k].role == "developer")) {
        k++;
    }

    common_chat_templates_inputs base = inputs;
    base.messages.assign(inputs.messages.begin(), inputs.messages.begin() + k);
    // a custom grammar never renders (and with tools it is refused); a json_schema can (DeepSeek
    // writes it into the system turn), so it stays
    base.grammar.clear();
    base.continue_final_message = COMMON_CHAT_CONTINUATION_NONE;
    base.add_generation_prompt  = true;

    auto msg = [](const char * role, const char * content) {
        common_chat_msg m;
        m.role    = role;
        m.content = content;
        return m;
    };
    // The placeholders start with characters of different classes (a letter, a newline, nothing at
    // all), so whatever the real conversation starts with, at least one of them differs from it at
    // its first character: the boundary stops where the conversation starts, not inside it. U+2063
    // (invisible separator) keeps a placeholder from completing a word or a token. The multi-turn
    // probe catches templates that render the preamble differently once the conversation has more
    // than one turn (e.g. tools listed before the last user message).
    const std::vector<std::vector<common_chat_msg>> probes = {
        { msg("user", "A\u2063") },
        { msg("user", "\n\u2063") },
        { msg("user", "") },
        { msg("user", "A\u2063"), msg("assistant", "B"), msg("user", "C") },
    };

    size_t end = std::string::npos;
    for (const auto & tail : probes) {
        common_chat_templates_inputs probe = base;
        probe.messages.insert(probe.messages.end(), tail.begin(), tail.end());
        std::string rendered;
        try {
            rendered = common_chat_templates_apply(tmpls, probe).prompt.str();
        } catch (const std::exception & e) {
            LOG_DBG("%s: probe not rendered: %s\n", __func__, e.what());
            continue;
        }
        size_t n = 0;
        while (n < prompt.size() && n < rendered.size() && prompt[n] == rendered[n]) {
            n++;
        }
        end = std::min(end, n);
    }
    return end == std::string::npos ? -1 : (int32_t) end;
}
