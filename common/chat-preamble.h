// fork-only: where the preamble (system context) of a rendered chat prompt ends, used by the server
// to place the disk cache's system-prompt node. Included by chat.h.

#pragma once

#include <cstdint>
#include <string>

struct common_chat_templates;
struct common_chat_templates_inputs;

// End of the preamble of a chat prompt, in characters: the longest prefix of `prompt` (the render of
// `inputs`) that is decided by the leading system / developer messages alone, whatever conversation
// follows them. Template-generic: the same messages are rendered again, each time followed by a
// different placeholder conversation (a user message starting with a letter, one starting with a
// newline, an empty one, and a user / assistant / user exchange), and the boundary is the shortest
// common prefix of `prompt` with those renders. Tools, template kwargs and the thinking flags are
// kept, since they render into the preamble, and so is a json_schema (some templates write it into
// the system turn); the continuation and a custom grammar are not. Returns -1 when no probe renders. A template that inserts a default system prompt gets a
// preamble even with no system message. Used to place the disk cache's system-prompt node.
int32_t common_chat_preamble_end(const struct common_chat_templates *        tmpls,
                                 const struct common_chat_templates_inputs & inputs,
                                 const std::string &                         prompt);
