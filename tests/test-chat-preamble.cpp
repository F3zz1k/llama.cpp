// Tests common_chat_preamble_end, the template-generic end of a chat prompt's system preamble that
// places the disk cache's system-prompt node, over every template in models/templates (plus any
// template files given on the command line).
//
// For each template, with and without tools, with and without a system message, the boundary must:
//   1. not depend on the conversation: the same for every first user message (letters, digits,
//      whitespace, newlines, punctuation, markup, CJK, emoji, empty), for a second turn, and for an
//      assistant prefill (continue_final_message) or a response_format schema;
//   2. be a prefix of every one of those prompts, so a node saved for one conversation restores for
//      the others;
//   3. for a request carrying only the system prompt (a pre-cache), be no longer than that and a
//      prefix of the system + user prompts.
// A template that raises on the probes must make the function return -1, not throw.

#include "chat.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static common_chat_msg msg(const std::string & role, const std::string & content) {
    common_chat_msg m;
    m.role    = role;
    m.content = content;
    return m;
}

struct rendered {
    std::string prompt;
    int32_t     end;
};

static std::optional<rendered> render(const common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & msgs,
                                      const std::vector<common_chat_tool> & tools,
                                      bool prefill, bool schema) {
    common_chat_templates_inputs in;
    in.messages  = msgs;
    in.tools     = tools;
    in.use_jinja = true;
    if (prefill) {
        in.add_generation_prompt  = false;
        in.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
    }
    if (schema) {
        in.json_schema = R"({"type":"object","properties":{"a":{"type":"string"}}})";
    }
    std::string prompt;
    try {
        prompt = common_chat_templates_apply(tmpls, in).prompt;
    } catch (const std::exception &) {
        return std::nullopt;
    }
    return rendered{ prompt, common_chat_preamble_end(tmpls, in, prompt) };
}

static int n_fail = 0;

static void fail(const std::string & name, const std::string & what) {
    fprintf(stderr, "FAIL %s: %s\n", name.c_str(), what.c_str());
    n_fail++;
}

// returns true when the template was exercised
static bool check_template(const std::string & name, const std::string & src, int & n_no_system_text) {
    common_chat_templates_ptr tmpls;
    try {
        tmpls = common_chat_templates_init(/* model= */ nullptr, src);
    } catch (const std::exception &) {
        return false;
    }

    const std::string sys = "You are a careful assistant. Keep answers short, cite your sources and never guess.";
    common_chat_tool tool;
    tool.name        = "get_weather";
    tool.description = "Get the weather for a city";
    tool.parameters  = R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})";

    const std::vector<std::string> firsts = {
        "", "Hello", "\n\ntwo newlines", " leading space", "1234", "<b>tag</b>",
        "\xe4\xb8\x80\xe4\xba\x8c\xe4\xb8\x89", "\xf0\x9f\x98\x80 hi",
    };

    bool exercised = false;
    for (int with_tools = 0; with_tools < 2; with_tools++) {
        const std::vector<common_chat_tool> tools = with_tools ? std::vector<common_chat_tool>{ tool } : std::vector<common_chat_tool>{};
        for (int with_sys = 0; with_sys < 2; with_sys++) {
            const std::string tag = name + (with_tools ? " +tools" : "") + (with_sys ? " +system" : " no-system");
            std::vector<common_chat_msg> base;
            if (with_sys) {
                base.push_back(msg("system", sys));
            }

            // a schema can render into the preamble (DeepSeek), so schema requests form their own group
            std::vector<std::pair<std::string, rendered>> all, with_schema;
            for (const auto & first : firsts) {
                auto m1 = base;
                m1.push_back(msg("user", first));
                if (auto r = render(tmpls.get(), m1, tools, false, false)) {
                    all.push_back({ "user '" + first + "'", *r });
                }
                // the other shapes for two first messages only (each render runs the template's parser
                // generation, and the autoparser templates are slow): empty, and a plain word
                if (first.size() > 5) {
                    continue;
                }
                auto m2 = m1;
                m2.push_back(msg("assistant", "Sure."));
                m2.push_back(msg("user", "And then?"));
                if (auto r = render(tmpls.get(), m2, tools, false, false)) {
                    all.push_back({ "two turns from '" + first + "'", *r });
                }
                auto m3 = m1;
                m3.push_back(msg("assistant", "Sure"));
                if (auto r = render(tmpls.get(), m3, tools, true, false)) {
                    all.push_back({ "prefill after '" + first + "'", *r });
                }
                if (!with_tools) {
                    if (auto r = render(tmpls.get(), m1, tools, false, true)) {
                        with_schema.push_back({ "schema, user '" + first + "'", *r });
                    }
                }
            }
            if (all.empty()) {
                continue;
            }
            exercised = true;

            const auto check_group = [&](const std::vector<std::pair<std::string, rendered>> & group) -> int32_t {
                if (group.empty()) {
                    return -1;
                }
                const int32_t c = group.front().second.end;
                if (c < 0) {
                    fail(tag, "no probe rendered for a template that renders the request");
                    return -1;
                }
                const std::string pre = group.front().second.prompt.substr(0, c);
                for (const auto & [what, r] : group) {
                    if (r.end != c) {
                        fail(tag, what + ": boundary " + std::to_string(r.end) + " differs from " + std::to_string(c));
                    }
                    if (r.prompt.compare(0, pre.size(), pre) != 0) {
                        fail(tag, what + ": the preamble is not a prefix of its prompt");
                    }
                }
                return c;
            };
            check_group(with_schema);
            const int32_t c = check_group(all);
            if (c < 0) {
                continue;
            }
            const std::string pre = all.front().second.prompt.substr(0, c);

            if (with_sys) {
                if (pre.find(sys) == std::string::npos) {
                    // legitimate for templates that move the system text into a later turn (e.g.
                    // Mistral-Nemo puts it in the last user message): no stable system prefix exists
                    n_no_system_text++;
                    printf("  note: %s: the system text is not inside the preamble (%d chars)\n", tag.c_str(), c);
                }
                if (auto rs = render(tmpls.get(), base, tools, false, false)) {
                    if (rs->end < 0) {
                        fail(tag, "system-only request: no probe rendered");
                    } else if (rs->end > c) {
                        fail(tag, "system-only boundary " + std::to_string(rs->end) + " is past the system + user one " + std::to_string(c));
                    } else if (pre.compare(0, rs->end, rs->prompt, 0, rs->end) != 0) {
                        fail(tag, "the system-only preamble is not a prefix of the system + user prompts");
                    }
                }
            }
        }
    }
    return exercised;
}

// --bench <template files...>: time one render of a request against the boundary probes, with 0,
// 10 and 50 tools (the cost a cache miss in server_preamble_cache adds before prefill)
static int bench(int argc, char ** argv) {
    for (int i = 2; i < argc; i++) {
        common_chat_templates_ptr tmpls = common_chat_templates_init(nullptr, read_file(argv[i]));
        for (int n_tools : { 0, 10, 50 }) {
            common_chat_templates_inputs in;
            in.use_jinja = true;
            in.messages  = { msg("system", std::string(4000, 'x').replace(0, 20, "You are a helper. ")), msg("user", "Hello") };
            for (int t = 0; t < n_tools; t++) {
                common_chat_tool tool;
                tool.name        = "tool_" + std::to_string(t);
                tool.description = "Does thing number " + std::to_string(t) + " to a file in the workspace.";
                tool.parameters  = R"({"type":"object","properties":{"path":{"type":"string"},"n":{"type":"integer"}},"required":["path"]})";
                in.tools.push_back(tool);
            }
            std::vector<double> t_apply, t_probe;
            for (int r = 0; r < 20; r++) {
                const auto t0 = std::chrono::steady_clock::now();
                std::string prompt;
                try {
                    prompt = common_chat_templates_apply(tmpls.get(), in).prompt;
                } catch (const std::exception &) {
                    break;
                }
                const auto t1 = std::chrono::steady_clock::now();
                common_chat_preamble_end(tmpls.get(), in, prompt);
                const auto t2 = std::chrono::steady_clock::now();
                t_apply.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                t_probe.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
            if (t_apply.empty()) {
                continue;
            }
            std::sort(t_apply.begin(), t_apply.end());
            std::sort(t_probe.begin(), t_probe.end());
            printf("%s tools=%d: apply p50 %.2f ms, probes p50 %.2f ms max %.2f ms\n", argv[i], n_tools,
                   t_apply[t_apply.size() / 2], t_probe[t_probe.size() / 2], t_probe.back());
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        return bench(argc, argv);
    }
    std::vector<std::string> files;
    for (const auto & e : std::filesystem::directory_iterator("models/templates")) {
        if (e.path().extension() == ".jinja") {
            files.push_back(e.path().string());
        }
    }
    for (int i = 1; i < argc; i++) {
        files.push_back(argv[i]);
    }
    std::sort(files.begin(), files.end());

    int n_checked = 0, n_skipped = 0, n_no_system_text = 0;
    for (const auto & f : files) {
        if (check_template(f, read_file(f), n_no_system_text)) {
            n_checked++;
        } else {
            n_skipped++;
            printf("  skipped (does not initialise or render): %s\n", f.c_str());
        }
    }

    // a template that raises on every probe: -1, never an exception
    {
        const std::string src =
            "{%- for m in messages %}"
            "{%- if m.role == 'user' and 'REAL' not in m.content %}{{ raise_exception('not a real message') }}{%- endif %}"
            "{{ '<' + m.role + '>' + m.content + '\\n' }}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}{{ '<assistant>' }}{%- endif %}";
        common_chat_templates_ptr tmpls;
        try {
            tmpls = common_chat_templates_init(nullptr, src);
        } catch (const std::exception & e) {
            fail("raising template", std::string("does not initialise: ") + e.what());
        }
        if (tmpls) {
            common_chat_templates_inputs in;
            in.messages  = { msg("system", "S"), msg("user", "REAL question") };
            in.use_jinja = true;
            try {
                const std::string prompt = common_chat_templates_apply(tmpls.get(), in).prompt;
                const int32_t c = common_chat_preamble_end(tmpls.get(), in, prompt);
                if (c != -1) {
                    fail("raising template", "expected -1, got " + std::to_string(c));
                }
            } catch (const std::exception & e) {
                fail("raising template", std::string("threw: ") + e.what());
            }
        }
    }

    printf("templates checked: %d, skipped: %d, system text outside the preamble: %d, failures: %d\n",
           n_checked, n_skipped, n_no_system_text, n_fail);
    if (n_checked < 50) {
        fail("coverage", "fewer than 50 templates exercised; the test is not looking at the template zoo");
    }
    return n_fail == 0 ? 0 : 1;
}
