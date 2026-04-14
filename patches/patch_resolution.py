#!/usr/bin/env python3
"""Patch run_test_pipeline.sh to accept --gen_width / --gen_height / --denoising_steps.
Idempotent — safe to re-run.
"""
import sys
import shutil
import os

SCRIPT = "/workspace/inspatio-world/run_test_pipeline.sh"
BACKUP = SCRIPT + ".orig"

with open(SCRIPT, 'r') as f:
    content = f.read()

if 'GEN_WIDTH' in content:
    print('Already patched, skipping.')
    sys.exit(0)

# Backup
if not os.path.exists(BACKUP):
    shutil.copy2(SCRIPT, BACKUP)

lines = content.split('\n')
new_lines = []

for i, line in enumerate(lines):
    # 1. Add defaults after COMPILE_DIT=false
    if line.strip() == 'COMPILE_DIT=false':
        new_lines.append(line)
        new_lines.append('GEN_WIDTH=832')
        new_lines.append('GEN_HEIGHT=480')
        new_lines.append('DENOISING_STEPS=""')
        continue

    # 2. Add arg cases before *) catch-all
    if line.strip() == '*)' and i + 1 < len(lines) and 'Unknown option' in lines[i + 1]:
        new_lines.append('        --gen_width)')
        new_lines.append('            GEN_WIDTH="$2"')
        new_lines.append('            shift 2')
        new_lines.append('            ;;')
        new_lines.append('        --gen_height)')
        new_lines.append('            GEN_HEIGHT="$2"')
        new_lines.append('            shift 2')
        new_lines.append('            ;;')
        new_lines.append('        --denoising_steps)')
        new_lines.append('            DENOISING_STEPS="$2"')
        new_lines.append('            shift 2')
        new_lines.append('            ;;')
        new_lines.append(line)
        continue

    # 3. Replace hardcoded DA3 resize
    if '"fix_resize_height":480,"fix_resize_width":832' in line:
        line = line.replace(
            '"fix_resize_height":480,"fix_resize_width":832',
            '"fix_resize_height":${GEN_HEIGHT},"fix_resize_width":${GEN_WIDTH}'
        )

    # 4. Replace hardcoded render dimensions
    if '--width 832 --height 480' in line:
        line = line.replace('--width 832 --height 480', '--width ${GEN_WIDTH} --height ${GEN_HEIGHT}')

    new_lines.append(line)

# 5. Add config override in Step 3 temp config section
# Find where TMP_CONFIG is created and sed commands are applied
final_lines = []
for i, line in enumerate(new_lines):
    final_lines.append(line)
    # After the freeze_repeat sed, add our overrides
    if 'freeze_repeat:.*|freeze_repeat:' in line and 'sed' in line:
        final_lines.append('')
        final_lines.append('    # Override resolution and denoising steps')
        final_lines.append('    sed -i "s/^\\(  \\)\\{0,1\\}video_size:.*//" "$TMP_CONFIG"')
        final_lines.append('    echo "" >> "$TMP_CONFIG"')
        final_lines.append('    echo "  video_size:" >> "$TMP_CONFIG"')
        final_lines.append('    echo "  - ${GEN_HEIGHT}" >> "$TMP_CONFIG"')
        final_lines.append('    echo "  - ${GEN_WIDTH}" >> "$TMP_CONFIG"')
        final_lines.append('    if [ -n "$DENOISING_STEPS" ]; then')
        final_lines.append('        sed -i "/^denoising_step_list:/,/^[^ ]/{ /^denoising_step_list:/d; /^- /d; }" "$TMP_CONFIG"')
        final_lines.append('        echo "denoising_step_list:" >> "$TMP_CONFIG"')
        final_lines.append('        IFS="," read -ra STEPS <<< "$DENOISING_STEPS"')
        final_lines.append('        for s in "${STEPS[@]}"; do')
        final_lines.append('            echo "- ${s}" >> "$TMP_CONFIG"')
        final_lines.append('        done')
        final_lines.append('    fi')

content = '\n'.join(final_lines)

with open(SCRIPT, 'w') as f:
    f.write(content)

print(f'Patched successfully. Added --gen_width, --gen_height, --denoising_steps.')
print(f'Backup saved to {BACKUP}')
