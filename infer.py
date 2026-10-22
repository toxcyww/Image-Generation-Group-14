"""Command line interface for the V3 four-column face anonymizer."""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description='V3: frozen models, baseline and gaze/chroma optimized image, 200 steps each')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--input', type=Path, help='Square face crop or directory of face crops')
    mode.add_argument('--latent', type=Path, help='Render a saved W+ latent without optimization')
    mode.add_argument('--check-models', action='store_true', help='Check required model paths and SHA256 hashes')
    parser.add_argument('--output', type=Path, help='A NEW output directory; a NEW PNG for --latent')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--seed', type=int, help='Optional full donor seed')
    parser.add_argument('--initial-latent', type=Path, help='Optional saved input W+ for exact-start replay')
    parser.add_argument('--teacher', type=Path, help='Optional saved teacher PNG; requires --teacher-latent')
    parser.add_argument('--teacher-latent', type=Path, help='The original latent that generated --teacher')
    args = parser.parse_args()
    if args.check_models:
        from anonymizer import check_models
        check_models()
        print('All six required pretrained files are present and SHA256 verified.')
        return 0
    if args.output is None:
        parser.error('--output is required for image/latent input')
    if args.output.exists():
        parser.error('Output exists; choose a new output path.')
    if bool(args.teacher) != bool(args.teacher_latent):
        parser.error('--teacher and --teacher-latent must be supplied together')
    if args.latent and any([args.teacher, args.teacher_latent, args.initial_latent, args.seed is not None]):
        parser.error('Teacher/seed/initial latent options apply only to image inputs')
    for name in ['input', 'latent', 'teacher', 'teacher_latent', 'initial_latent']:
        path = getattr(args, name)
        if path is not None and not path.exists():
            parser.error(name+' path does not exist')
    images = None
    if args.input and args.input.is_dir():
        if args.teacher or args.initial_latent:
            parser.error('Saved-start/teacher options support a single image only')
        images = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'})
        if not images:
            parser.error('Input directory has no supported images')
        if len({p.stem.casefold() for p in images}) != len(images):
            parser.error('Batch image filenames must have distinct stems')
    from anonymizer import Anonymizer, write_json
    model = Anonymizer(args.device)
    if args.latent:
        model.render_latent(args.latent, args.output)
    elif images is None:
        model.anonymize(args.input, args.output, seed=args.seed, initial_latent=args.initial_latent,
                        teacher_path=args.teacher, teacher_latent=args.teacher_latent)
    else:
        args.output.mkdir(parents=True)
        records = []
        for image in images:
            try:
                result = model.anonymize(image, args.output/image.stem, seed=args.seed)
                records.append({'input': image.name, 'status': result['status']})
            except Exception as error:
                records.append({'input': image.name, 'status': 'failed', 'error': type(error).__name__, 'message': str(error)})
            write_json(args.output/'batch_results.json', records)
        if any(r['status'] != 'completed' for r in records):
            print('Some images failed; see batch_results.json', file=sys.stderr)
            return 1
    print('Saved:', args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
