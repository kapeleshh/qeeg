"""
Report generation module for EEG data visualization.

This module provides functions for generating comprehensive reports from EEG data,
including spectral analysis, asymmetry analysis, and clinical findings.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import matplotlib.pyplot as plt
import mne
import os
import datetime


def generate_spectral_report(
    raw: mne.io.Raw,
    output_dir: str = "reports",
    filename: Optional[str] = None,
    show: bool = False,
    **kwargs
) -> str:
    """
    Generate a report with spectral analysis of EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    output_dir : str, optional
        Directory to save the report, by default "reports"
    filename : str or None, optional
        Filename for the report, by default None (auto-generated)
    show : bool, optional
        Whether to show the figures during generation, by default False
    **kwargs
        Additional keyword arguments to pass to plotting functions

    Returns
    -------
    str
        Path to the generated report.

    Examples
    --------
    >>> import mne
    >>> from qeeg.visualization import reports
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report_path = reports.generate_spectral_report(raw)
    >>> print(f"Report saved to: {report_path}")
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subject = raw.info.get('subject_info', {}).get('his_id', 'unknown')
        filename = f"spectral_report_{subject}_{timestamp}.html"
    
    # Full path to the report
    report_path = os.path.join(output_dir, filename)
    
    # Create an MNE Report
    report = mne.Report(title="Spectral Analysis Report", verbose=False)
    
    # Add information about the recording
    info_html = f"""
    <h2>Recording Information</h2>
    <table>
        <tr><td>Subject:</td><td>{raw.info.get('subject_info', {}).get('his_id', 'Unknown')}</td></tr>
        <tr><td>Recording Date:</td><td>{raw.info.get('meas_date', 'Unknown')}</td></tr>
        <tr><td>Sampling Rate:</td><td>{raw.info['sfreq']} Hz</td></tr>
        <tr><td>Duration:</td><td>{raw.times[-1]:.2f} seconds</td></tr>
        <tr><td>Number of Channels:</td><td>{len(raw.ch_names)}</td></tr>
    </table>
    """
    report.add_html(info_html, title="Recording Information")
    
    # Add a power spectrum plot
    from qeeg.visualization import spectra
    fig_psd = spectra.plot_power_spectrum(raw, show=show, **kwargs)
    report.add_figure(fig_psd, title="Power Spectrum", caption="Power spectral density of EEG channels.")
    plt.close(fig_psd)
    
    # Add a band power comparison plot
    fig_bands = spectra.plot_band_power_comparison(raw, show=show, **kwargs)
    report.add_figure(fig_bands, title="Frequency Band Powers", caption="Comparison of power in different frequency bands.")
    plt.close(fig_bands)
    
    # Add topographic maps
    from qeeg.visualization import topomaps
    fig_topo = topomaps.plot_band_topomaps(raw, show=show, **kwargs)
    report.add_figure(fig_topo, title="Topographic Maps", caption="Topographic distribution of power in different frequency bands.")
    plt.close(fig_topo)
    
    # Save the report
    report.save(report_path, overwrite=True, open_browser=False)
    
    return report_path


def generate_clinical_report(
    raw: mne.io.Raw,
    conditions: List[str] = ["adhd", "anxiety", "depression", "autism"],
    output_dir: str = "reports",
    filename: Optional[str] = None,
    show: bool = False,
    **kwargs
) -> str:
    """
    Generate a clinical report with analysis of EEG data for various conditions.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    conditions : List[str], optional
        List of conditions to analyze, by default ["adhd", "anxiety", "depression", "autism"]
    output_dir : str, optional
        Directory to save the report, by default "reports"
    filename : str or None, optional
        Filename for the report, by default None (auto-generated)
    show : bool, optional
        Whether to show the figures during generation, by default False
    **kwargs
        Additional keyword arguments to pass to analysis functions

    Returns
    -------
    str
        Path to the generated report.

    Examples
    --------
    >>> import mne
    >>> from qeeg.visualization import reports
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report_path = reports.generate_clinical_report(raw)
    >>> print(f"Report saved to: {report_path}")
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subject = raw.info.get('subject_info', {}).get('his_id', 'unknown')
        filename = f"clinical_report_{subject}_{timestamp}.html"
    
    # Full path to the report
    report_path = os.path.join(output_dir, filename)
    
    # Create an MNE Report
    report = mne.Report(title="Clinical EEG Analysis Report", verbose=False)
    
    # Add information about the recording
    info_html = f"""
    <h2>Recording Information</h2>
    <table>
        <tr><td>Subject:</td><td>{raw.info.get('subject_info', {}).get('his_id', 'Unknown')}</td></tr>
        <tr><td>Recording Date:</td><td>{raw.info.get('meas_date', 'Unknown')}</td></tr>
        <tr><td>Sampling Rate:</td><td>{raw.info['sfreq']} Hz</td></tr>
        <tr><td>Duration:</td><td>{raw.times[-1]:.2f} seconds</td></tr>
        <tr><td>Number of Channels:</td><td>{len(raw.ch_names)}</td></tr>
    </table>
    """
    report.add_html(info_html, title="Recording Information")
    
    # Add spectral analysis
    from qeeg.visualization import spectra
    fig_psd = spectra.plot_power_spectrum(raw, show=show, **kwargs)
    report.add_figure(fig_psd, title="Power Spectrum", caption="Power spectral density of EEG channels.")
    plt.close(fig_psd)
    
    # Add topographic maps
    from qeeg.visualization import topomaps
    fig_topo = topomaps.plot_band_topomaps(raw, show=show, **kwargs)
    report.add_figure(fig_topo, title="Topographic Maps", caption="Topographic distribution of power in different frequency bands.")
    plt.close(fig_topo)
    
    # Add asymmetry analysis
    fig_asymm = topomaps.plot_asymmetry_topomap(raw, band='Alpha', show=show, **kwargs)
    report.add_figure(fig_asymm, title="Alpha Asymmetry", caption="Topographic distribution of alpha asymmetry.")
    plt.close(fig_asymm)
    
    # Add condition-specific analyses
    for condition in conditions:
        if condition == "adhd":
            from qeeg.conditions import adhd
            adhd_report = adhd.generate_adhd_report(raw, **kwargs)
            report.add_html(f"<pre>{adhd_report}</pre>", title="ADHD Analysis")
        
        elif condition == "anxiety":
            from qeeg.conditions import anxiety
            anxiety_report = anxiety.generate_anxiety_report(raw, **kwargs)
            report.add_html(f"<pre>{anxiety_report}</pre>", title="Anxiety Analysis")
        
        elif condition == "depression":
            from qeeg.conditions import depression
            depression_report = depression.generate_depression_report(raw, **kwargs)
            report.add_html(f"<pre>{depression_report}</pre>", title="Depression Analysis")
        
        elif condition == "autism":
            from qeeg.conditions import autism
            autism_report = autism.generate_autism_report(raw, **kwargs)
            report.add_html(f"<pre>{autism_report}</pre>", title="Autism Analysis")
        
        elif condition in ["insomnia", "tbi", "alzheimers", "parkinsons", "schizophrenia"]:
            from qeeg.conditions import other_conditions
            other_report = other_conditions.generate_condition_report(raw, condition, **kwargs)
            report.add_html(f"<pre>{other_report}</pre>", title=f"{condition.capitalize()} Analysis")
    
    # Add a disclaimer
    disclaimer_html = """
    <div style="border: 1px solid #f00; padding: 10px; margin: 20px 0; background-color: #fff8f8;">
        <h3 style="color: #f00;">Disclaimer</h3>
        <p>This report is generated for research purposes only and should not be used for clinical diagnosis.
        The analyses and findings presented in this report are based on automated algorithms and should be
        interpreted by qualified healthcare professionals in conjunction with other clinical information.</p>
    </div>
    """
    report.add_html(disclaimer_html, title="Disclaimer")
    
    # Save the report
    report.save(report_path, overwrite=True, open_browser=False)
    
    return report_path


def generate_epileptiform_report(
    raw: mne.io.Raw,
    output_dir: str = "reports",
    filename: Optional[str] = None,
    show: bool = False,
    **kwargs
) -> str:
    """
    Generate a report focused on epileptiform activity in EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    output_dir : str, optional
        Directory to save the report, by default "reports"
    filename : str or None, optional
        Filename for the report, by default None (auto-generated)
    show : bool, optional
        Whether to show the figures during generation, by default False
    **kwargs
        Additional keyword arguments to pass to analysis functions

    Returns
    -------
    str
        Path to the generated report.

    Examples
    --------
    >>> import mne
    >>> from qeeg.visualization import reports
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report_path = reports.generate_epileptiform_report(raw)
    >>> print(f"Report saved to: {report_path}")
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subject = raw.info.get('subject_info', {}).get('his_id', 'unknown')
        filename = f"epileptiform_report_{subject}_{timestamp}.html"
    
    # Full path to the report
    report_path = os.path.join(output_dir, filename)
    
    # Create an MNE Report
    report = mne.Report(title="Epileptiform Activity Report", verbose=False)
    
    # Add information about the recording
    info_html = f"""
    <h2>Recording Information</h2>
    <table>
        <tr><td>Subject:</td><td>{raw.info.get('subject_info', {}).get('his_id', 'Unknown')}</td></tr>
        <tr><td>Recording Date:</td><td>{raw.info.get('meas_date', 'Unknown')}</td></tr>
        <tr><td>Sampling Rate:</td><td>{raw.info['sfreq']} Hz</td></tr>
        <tr><td>Duration:</td><td>{raw.times[-1]:.2f} seconds</td></tr>
        <tr><td>Number of Channels:</td><td>{len(raw.ch_names)}</td></tr>
    </table>
    """
    report.add_html(info_html, title="Recording Information")
    
    # Detect epileptiform activity
    from qeeg.analysis import epileptiform
    activities = epileptiform.detect_epileptiform_activity(raw, **kwargs)
    summary = epileptiform.analyze_epileptiform_activity(raw, **kwargs)
    
    # Add summary of epileptiform activity
    summary_html = "<h2>Summary of Epileptiform Activity</h2><table>"
    for activity_type, activity_summary in summary.items():
        summary_html += f"<tr><td><b>{activity_type.capitalize()}</b></td><td>Count: {activity_summary['count']}</td>"
        summary_html += f"<td>Rate: {activity_summary['rate']:.2f} per minute</td></tr>"
    summary_html += "</table>"
    
    report.add_html(summary_html, title="Epileptiform Activity Summary")
    
    # Add spike visualization if spikes were detected
    if activities['spikes']:
        # Plot the spikes on a topographic map
        from qeeg.visualization import topomaps
        fig_spikes = topomaps.plot_epileptiform_activity(raw, activities['spikes'], show=show, **kwargs)
        report.add_figure(fig_spikes, title="Spike Distribution", caption="Topographic distribution of detected spikes.")
        plt.close(fig_spikes)
        
        # Add a table with spike details
        spikes_html = "<h2>Detected Spikes</h2><table border='1'><tr><th>Channel</th><th>Time (s)</th><th>Duration (s)</th><th>Amplitude</th></tr>"
        for i, spike in enumerate(activities['spikes'][:20]):  # Limit to first 20 spikes
            spikes_html += f"<tr><td>{spike['channel']}</td><td>{spike['onset']:.2f}</td><td>{spike['duration']:.3f}</td><td>{spike['amplitude']:.2f}</td></tr>"
        if len(activities['spikes']) > 20:
            spikes_html += f"<tr><td colspan='4'>... and {len(activities['spikes']) - 20} more</td></tr>"
        spikes_html += "</table>"
        report.add_html(spikes_html, title="Spike Details")
    
    # Add OIRDA visualization if detected
    if activities['oirda']:
        oirda_html = "<h2>Occipital Intermittent Rhythmic Delta Activity (OIRDA)</h2><table border='1'><tr><th>Zone</th><th>Time (s)</th><th>Duration (s)</th><th>Power</th></tr>"
        for i, event in enumerate(activities['oirda'][:10]):  # Limit to first 10 events
            oirda_html += f"<tr><td>{event['zone']}</td><td>{event['onset']:.2f}</td><td>{event['duration']:.2f}</td><td>{event['power']:.2f}</td></tr>"
        if len(activities['oirda']) > 10:
            oirda_html += f"<tr><td colspan='4'>... and {len(activities['oirda']) - 10} more</td></tr>"
        oirda_html += "</table>"
        report.add_html(oirda_html, title="OIRDA Details")
    
    # Add FIRDA visualization if detected
    if activities['firda']:
        firda_html = "<h2>Frontal Intermittent Rhythmic Delta Activity (FIRDA)</h2><table border='1'><tr><th>Zone</th><th>Time (s)</th><th>Duration (s)</th><th>Power</th></tr>"
        for i, event in enumerate(activities['firda'][:10]):  # Limit to first 10 events
            firda_html += f"<tr><td>{event['zone']}</td><td>{event['onset']:.2f}</td><td>{event['duration']:.2f}</td><td>{event['power']:.2f}</td></tr>"
        if len(activities['firda']) > 10:
            firda_html += f"<tr><td colspan='4'>... and {len(activities['firda']) - 10} more</td></tr>"
        firda_html += "</table>"
        report.add_html(firda_html, title="FIRDA Details")
    
    # Add a disclaimer
    disclaimer_html = """
    <div style="border: 1px solid #f00; padding: 10px; margin: 20px 0; background-color: #fff8f8;">
        <h3 style="color: #f00;">Disclaimer</h3>
        <p>This report is generated for research purposes only and should not be used for clinical diagnosis.
        The analyses and findings presented in this report are based on automated algorithms and should be
        interpreted by qualified healthcare professionals in conjunction with other clinical information.</p>
    </div>
    """
    report.add_html(disclaimer_html, title="Disclaimer")
    
    # Save the report
    report.save(report_path, overwrite=True, open_browser=False)
    
    return report_path
