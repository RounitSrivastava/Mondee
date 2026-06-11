"""
scripts/build_index.py

Builds Hybrid FAISS Index

Sources:
- Amazon Metadata
- Amazon Reviews
- Custom Experience Catalogs

Supports:
- Amazon Metadata Format
  {'asin': '...', ...}
- JSON Lines
- JSON Arrays

Output:
- RoBERTa/BGE + LDA Hybrid Index
"""

import os
import sys
import glob
import re
import ast
import json

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        '..'
    )
)

from core.indexer import build_index

# =====================================================
# CONFIG
# =====================================================

MAX_RECORDS = 50000

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data'
)

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    '..',
    'index'
)

CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(__file__),
    '..',
    'checkpoints'
)

EXPERIENCES_CKPT = os.path.join(
    CHECKPOINTS_DIR,
    'experiences_checkpoint.json'
)

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if not text:
        return ''

    text = str(text)

    text = re.sub(
        r'<[^>]+>',
        ' ',
        text
    )

    text = text.lower()

    text = re.sub(
        r'[^a-z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


def load_checkpoint():

    if not os.path.exists(
        EXPERIENCES_CKPT
    ):

        return None

    print(
        f'\n[build_index] Loading checkpoint: '
        f'{EXPERIENCES_CKPT}'
    )

    with open(
        EXPERIENCES_CKPT,
        'r',
        encoding='utf-8'
    ) as f:

        data = json.load(f)

    print(
        f'[build_index] Restored '
        f'{len(data.get("experiences", []))} experiences'
    )

    return data.get('experiences', [])


def save_checkpoint(
    experiences: list[dict]
):

    os.makedirs(
        CHECKPOINTS_DIR,
        exist_ok=True
    )

    with open(
        EXPERIENCES_CKPT,
        'w',
        encoding='utf-8'
    ) as f:

        json.dump(
            {
                'experiences': experiences
            },
            f
        )

    print(
        f'[build_index] Checkpoint saved '
        f'({len(experiences)} experiences)'
    )


def remove_checkpoint():

    if os.path.exists(
        EXPERIENCES_CKPT
    ):

        os.remove(
            EXPERIENCES_CKPT
        )

        print(
            '[build_index] Checkpoint removed'
        )


# =====================================================
# BUILD DESCRIPTION
# =====================================================

def build_description(item):

    pieces = []

    description = item.get(
        'description',
        ''
    )

    summary = item.get(
        'summary',
        ''
    )

    brand = item.get(
        'brand',
        ''
    )

    categories = item.get(
        'categories',
        []
    )

    features = item.get(
        'feature',
        []
    )

    # Description

    if description:

        if isinstance(
            description,
            list
        ):

            pieces.extend(
                description
            )

        else:

            pieces.append(
                str(description)
            )

    # Summary

    if summary:

        pieces.append(
            str(summary)
        )

    # Brand

    if brand:

        pieces.append(
            str(brand)
        )

    # Categories

    if categories:

        for cat in categories:

            if isinstance(
                cat,
                list
            ):

                pieces.extend(cat)

            else:

                pieces.append(
                    str(cat)
                )

    # Features

    if features:

        if isinstance(
            features,
            list
        ):

            pieces.extend(features)

        else:

            pieces.append(
                str(features)
            )

    text = ' '.join(
        str(x)
        for x in pieces
    )

    return clean_text(text)


# =====================================================
# MAIN
# =====================================================

def main():

    files = glob.glob(
        os.path.join(
            DATA_DIR,
            '*.json'
        )
    )

    print('\nFound files:')

    for f in files:

        print(
            ' -',
            os.path.basename(f)
        )

    if not files:

        raise FileNotFoundError(
            'No JSON files found in /data'
        )

    experiences = load_checkpoint()

    if experiences is None:

        experiences = []

        seen_asins = set()

        total_raw = 0

        total_valid = 0

        # READ FILES
    
        for file in files:

            print(
                f'\n[build_index] Loading '
                f'{os.path.basename(file)}'
            )

            with open(
                file,
                'r',
                encoding='utf-8'
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:

                        continue

                    try:

                        item = ast.literal_eval(
                            line
                        )

                    except Exception:

                        continue

                    total_raw += 1

                    asin = str(
                        item.get(
                            'asin',
                            ''
                        )
                    ).strip()

                    if not asin:

                        continue

                    if asin in seen_asins:

                        continue

                    title = (

                        item.get(
                            'title',
                            ''
                        )

                        or

                        item.get(
                            'name',
                            ''
                        )

                        or

                        item.get(
                            'summary',
                            ''
                        )

                    )

                    if not title:

                        continue

                    details = build_description(
                        item
                    )

                    experiences.append({

                        'exp_id':
                            f'exp_{asin}',

                        'name':
                            clean_text(title),

                        'details':
                            details[:2000]
                    })

                    seen_asins.add(
                        asin
                    )

                    total_valid += 1

                    # Show first few samples

                    if total_valid <= 3:

                        print(
                            f'\nSample #{total_valid}'
                        )

                        print(
                            'ASIN:',
                            asin
                        )

                        print(
                            'TITLE:',
                            title[:100]
                        )

                    if len(
                        experiences
                    ) >= MAX_RECORDS:

                        break

                if len(
                    experiences
                ) >= MAX_RECORDS:

                    break

        save_checkpoint(
            experiences
        )

    else:

        total_raw = 0
        total_valid = len(experiences)
        seen_asins = {
            e.get('exp_id', '').replace('exp_', '')
            for e in experiences
        }
        print(
            f'\n[build_index] Skipped re-parsing data files '
            f'(using checkpoint with {len(experiences)} records)'
        )
    
    # SUMMARY
    

    print('\n' + '=' * 60)

    print(
        f'Raw records      : {total_raw}'
    )

    print(
        f'Unique products  : {len(experiences)}'
    )

    print(
        f'Max records      : {MAX_RECORDS}'
    )

    print('=' * 60)

    if not experiences:

        raise ValueError(
            'No valid experiences found.'
        )

    
    # BUILD INDEX
    
    build_index(
        experiences
    )

    # Note: Checkpoints are preserved for PR sharing per user request.
    # remove_checkpoint()
    print('\n[build_index] Checkpoint preserved in checkpoints/ directory.')

    print(
        '\n[build_index] ✓ Index Built'
    )


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == '__main__':

    main()