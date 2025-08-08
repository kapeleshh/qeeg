"""
Command-line interface commands for the qeeg package.

This module provides command-line tools for working with EEG data.
"""

import argparse
import sys
import mne
import numpy as np
from typing import List, Dict, Any, Optional, Union

from qeeg.preprocessing import filtering, artifacts
from qeeg.analysis import spectral, asymmetry, epileptiform


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    
    Returns
    -------
    argparse.ArgumentParser
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        description="QEEG: Quantitative EEG Analysis Toolkit",
        epilog="For more information, visit https://github.com/kapeleshh/qeeg"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess EEG data")
    preprocess_parser.add_argument("input", help="Input EEG file")
    preprocess_parser.add_argument("output", help="Output EEG file")
    preprocess_parser.add_argument("--filter", action="store_true", help="Apply bandpass filter")
    preprocess_parser.add_argument("--l-freq", type=float, default=1.0, help="Lower frequency bound for filtering")
    preprocess_parser.add_argument("--h-freq", type=float, default=40.0, help="Upper frequency bound for filtering")
    preprocess_parser.add_argument("--notch", action="store_true", help="Apply notch filter")
    preprocess_parser.add_argument("--notch-freq", type=float, default=50.0, help="Frequency to filter out with notch filter")
    preprocess_parser.add_argument("--ica", action="store_true", help="Apply ICA for artifact removal")
    preprocess_parser.add_argument("--reference", default="average", help="Reference for EEG data")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze EEG data")
    analyze_parser.add_argument("input", help="Input EEG file")
    analyze_parser.add_argument("--output", help="Output file for analysis results")
    analyze_parser.add_argument("--spectral", action="store_true", help="Perform spectral analysis")
    analyze_parser.add_argument("--asymmetry", action="store_true", help="Perform asymmetry analysis")
    analyze_parser.add_argument("--epileptiform", action="store_true", help="Detect epileptiform activity")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize EEG data")
    visualize_parser.add_argument("input", help="Input EEG file")
    visualize_parser.add_argument("--output", help="Output file for visualization")
    visualize_parser.add_argument("--type", choices=["raw", "psd", "topomap", "bands"], default="raw", help="Type of visualization")
    
    return parser


def preprocess_command(args: argparse.Namespace) -> None:
    """
    Execute the preprocess command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Loading EEG data from {args.input}...")
    raw = mne.io.read_raw(args.input, preload=True)
    
    if args.filter:
        print(f"Applying bandpass filter ({args.l_freq} - {args.h_freq} Hz)...")
        raw = filtering.bandpass_filter(raw, l_freq=args.l_freq, h_freq=args.h_freq)
    
    if args.notch:
        print(f"Applying notch filter at {args.notch_freq} Hz...")
        raw = filtering.notch_filter(raw, freqs=args.notch_freq)
    
    if args.ica:
        print("Applying ICA for artifact removal...")
        raw, _ = artifacts.apply_ica(raw)
    
    if args.reference:
        print(f"Setting reference to {args.reference}...")
        raw.set_eeg_reference(args.reference, projection=True)
    
    print(f"Saving preprocessed data to {args.output}...")
    raw.save(args.output, overwrite=True)
    print("Done!")


def analyze_command(args: argparse.Namespace) -> None:
    """
    Execute the analyze command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Loading EEG data from {args.input}...")
    raw = mne.io.read_raw(args.input, preload=True)
    
    results = {}
    
    if args.spectral:
        print("Performing spectral analysis...")
        psds, freqs = spectral.compute_psd(raw)
        band_powers = spectral.compute_band_powers(raw)
        results["spectral"] = {
            "psds": psds.tolist(),
            "freqs": freqs.tolist(),
            "band_powers": {band: powers.tolist() for band, powers in band_powers.items()}
        }
    
    if args.asymmetry:
        print("Performing asymmetry analysis...")
        asymmetry_indices = asymmetry.compute_asymmetry_index(raw)
        results["asymmetry"] = {pair: float(index) for pair, index in asymmetry_indices.items()}
    
    if args.epileptiform:
        print("Detecting epileptiform activity...")
        activities = epileptiform.detect_epileptiform_activity(raw)
        results["epileptiform"] = {
            activity_type: [
                {k: v for k, v in event.items() if k != "channel" or isinstance(v, str)}
                for event in events
            ]
            for activity_type, events in activities.items()
        }
    
    if args.output:
        import json
        print(f"Saving analysis results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    else:
        import json
        print(json.dumps(results, indent=2))
    
    print("Done!")


def visualize_command(args: argparse.Namespace) -> None:
    """
    Execute the visualize command.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print(f"Loading EEG data from {args.input}...")
    raw = mne.io.read_raw(args.input, preload=True)
    
    import matplotlib.pyplot as plt
    
    if args.type == "raw":
        print("Visualizing raw EEG data...")
        fig = raw.plot(show=False)
    elif args.type == "psd":
        print("Visualizing power spectral density...")
        fig = raw.plot_psd(show=False)
    elif args.type == "topomap":
        print("Visualizing topographic map...")
        # Create evoked data for topomap
        evoked = mne.EvokedArray(raw.get_data()[:, :1000].mean(axis=1)[:, np.newaxis],
                                raw.info, tmin=0)
        fig = evoked.plot_topomap(show=False)
    elif args.type == "bands":
        print("Visualizing frequency bands...")
        from qeeg.visualization import topomaps
        band_powers = spectral.compute_band_powers(raw)
        fig = topomaps.plot_band_topomaps(band_powers, raw.info)
    
    if args.output:
        print(f"Saving visualization to {args.output}...")
        fig.savefig(args.output)
    else:
        plt.show()
    
    print("Done!")


def main() -> None:
    """
    Main entry point for the command-line interface.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "visualize":
        visualize_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
