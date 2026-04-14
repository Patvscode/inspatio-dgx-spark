#!/bin/bash
# Patch run_test_pipeline.sh inside the container to accept --gen_width / --gen_height / --denoising_steps
# Idempotent — safe to re-run. Saves backup on first run.
set -e

CONTAINER="inspatio-world"
SCRIPT="/workspace/inspatio-world/run_test_pipeline.sh"
BACKUP="/workspace/inspatio-world/run_test_pipeline.sh.orig"

# Backup original (only once)
docker exec "$CONTAINER" bash -c "[ ! -f '$BACKUP' ] && cp '$SCRIPT' '$BACKUP' || true"

# Write the patch as a sed script inside the container
docker exec "$CONTAINER" bash -c "cat > /tmp/patch_resolution.py << 'PYEOF'
import re, sys

with open('$SCRIPT', 'r') as f:
    content = f.read()

# Skip if already patched
if 'GEN_WIDTH' in content:
    print('Already patched, skipping.')
    sys.exit(0)

# 1. Add default vars after COMPILE_DIT=false
content = content.replace(
    'COMPILE_DIT=false',
    'COMPILE_DIT=false\nGEN_WIDTH=832\nGEN_HEIGHT=480\nDENOISING_STEPS=\"\"'
)

# 2. Add arg parsing cases before the *) catch-all
new_cases = '''        --gen_width)
            GEN_WIDTH=\"\$2\"
            shift 2
            ;;
        --gen_height)
            GEN_HEIGHT=\"\$2\"
            shift 2
            ;;
        --denoising_steps)
            DENOISING_STEPS=\"\$2\"
            shift 2
            ;;
'''
content = content.replace(
    '        *)\\n            echo \"Unknown option: \$1\"',
    new_cases + '        *)\\n            echo \"Unknown option: \$1\"'
)
# Try alternate match (without backslash-n)
if '--gen_width' not in content:
    lines = content.split('\\n')
    new_lines = []
    for i, line in enumerate(lines):
        if line.strip() == '*)' and i > 0 and 'Unknown option' in (lines[i+1] if i+1 < len(lines) else ''):
            new_lines.append('        --gen_width)')
            new_lines.append('            GEN_WIDTH=\"\$2\"')
            new_lines.append('            shift 2')
            new_lines.append('            ;;')
            new_lines.append('        --gen_height)')
            new_lines.append('            GEN_HEIGHT=\"\$2\"')
            new_lines.append('            shift 2')
            new_lines.append('            ;;')
            new_lines.append('        --denoising_steps)')
            new_lines.append('            DENOISING_STEPS=\"\$2\"')
            new_lines.append('            shift 2')
            new_lines.append('            ;;')
        new_lines.append(line)
    content = '\\n'.join(new_lines)

# 3. Replace hardcoded DA3 resize dimensions
content = content.replace(
    '\"fix_resize_height\":480,\"fix_resize_width\":832',
    '\"fix_resize_height\":\${GEN_HEIGHT},\"fix_resize_width\":\${GEN_WIDTH}'
)

# 4. Replace hardcoded render dimensions
content = content.replace(
    '--width 832 --height 480',
    '--width \${GEN_WIDTH} --height \${GEN_HEIGHT}'
)

# 5. Add video_size override in the temp config section (after the sed commands for traj)
# Find the line that does freeze_frame sed and add video_size + denoising override after it
old_marker = 'sed -i \"/^[[:space:]]*#/!s|freeze_repeat:.*|freeze_repeat: \${FREEZE_REPEAT}|g\" \"\$TMP_CONFIG\"'
new_addition = old_marker + '''

    # Override video_size for resolution control
    sed -i \"/^[[:space:]]*#/!s|video_size:|video_size:|g\" \"\$TMP_CONFIG\"
    python3 -c \"
import yaml, sys
with open('\$TMP_CONFIG', 'r') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('dataset', {})['video_size'] = [\${GEN_HEIGHT}, \${GEN_WIDTH}]
if '\${DENOISING_STEPS}':
    steps = [int(x) for x in '\${DENOISING_STEPS}'.split(',') if x.strip()]
    if steps:
        cfg['denoising_step_list'] = steps
with open('\$TMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
\" 2>/dev/null || echo \"Warning: Could not override video_size in config (yaml module missing?)\"
'''
if old_marker in content:
    content = content.replace(old_marker, new_addition)

with open('$SCRIPT', 'w') as f:
    f.write(content)

print('Patched successfully: --gen_width, --gen_height, --denoising_steps now accepted.')
PYEOF
python3 /tmp/patch_resolution.py"

echo "Patch applied."
