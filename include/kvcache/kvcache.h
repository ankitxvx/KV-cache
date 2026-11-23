#pragma once

/// @file kvcache.h
/// @brief Main public include header for the PageForge KV Cache library.
/// Include this single header to get the full KV Cache API.

// ── Core (v1) ──
#include "types.h"
#include "block_allocator.h"
#include "block_table.h"
#include "sequence.h"
#include "eviction_policy.h"
#include "quantization.h"
#include "prefix_cache.h"
#include "cache_manager.h"

// ── Production (v2) ──
#include "logger.h"
#include "metrics.h"
#include "config.h"
#include "gpu_allocator.h"
#include "ssd_allocator.h"
#include "tier_manager.h"
#include "transfer_engine.h"
#include "attention.h"
#include "scheduler.h"
