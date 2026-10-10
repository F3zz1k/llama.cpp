// fork-only: the command-line options of the auto disk prompt/KV cache (--slot-save-*), kept out of
// arg.cpp. common_params_parser_init and common_params_parse_ex each call in with a single line.

#pragma once

#include "arg.h"

#include <functional>

// registers the --slot-save-* options (and --slot-restore-*) through common_params_parser_init's add_opt
void common_params_add_disk_cache_args(const std::function<void(common_arg)> & add_opt, common_params & params);

// cross-option checks of those options, run by common_params_parse_ex after every option is parsed
void common_params_validate_disk_cache(const common_params & params);
