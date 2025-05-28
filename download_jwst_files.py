#!/usr/bin/env python3

import os
import sys
import argparse
import requests
import xml.etree.ElementTree as ET
from astroquery.mast import Observations


# Auto-disable colors if stdout is redirected
USE_COLOR = sys.stdout.isatty()


def color(code):
    return code if USE_COLOR else ""


# ANSI color codes (conditionally applied)
RESET = color("\033[0m")
GREEN = color("\033[92m")
RED = color("\033[91m")
YELLOW = color("\033[93m")
CYAN = color("\033[96m")

TAG_WIDTH = 13  # Space for the longest tag like "[DOWNLOADING]"


def log(tag, color_code, message):
    padded_tag = f"{color_code}{tag:<{TAG_WIDTH}}{RESET}"
    print(f"{padded_tag} {message}", flush=True)


def normalize_target(target):
    if target is None:
        return "UnknownTarget"
    if target.endswith("b"):
        return target
    elif target.endswith("B"):
        return target[:-1] + "b"
    else:
        return target + "b"


def query_MAST(proposal_id, observation_id, visit_id, subgroup='UNCAL'):
    # The following code will convert your inputs to properly formatted strings
    if type(proposal_id) is not str:
        proposal_id = str(proposal_id).zfill(5)
    if type(observation_id) is not str:
        observation_id = str(observation_id).zfill(3)
    if type(visit_id) is not str:
        visit_id = str(visit_id).zfill(3)

    if subgroup in ['UNCAL', 'GS-ACQ1', 'GS-ACQ2', 'GS-FG', 'GS-ID',
                    'GS-TRACK']:
        calib_level = [1,]
    elif subgroup in ['CAL', 'CALINTS', 'RATE', 'RATEINTS', 'X1DINTS',
                      'ANNNN_CRFINTS', 'GS-ACQ1', 'GS-ACQ2', 'GS-FG',
                      'GS-ID', 'GS-TRACK', 'RAMP']:
        calib_level = [2,]
    elif subgroup in ['X1DINTS', 'WHTLT']:
        calib_level = [3,]

    # This code will specify the obsid using wildcards
    obs_id = f'jw{proposal_id}{observation_id}{visit_id}_03101*'

    # Query MAST for requested visit
    sci_table = Observations.query_criteria(proposal_id=proposal_id,
                                            obs_id=obs_id)
    table = []
    if len(sci_table) > 0:
        # Get product list
        data_products_by_id = Observations.get_product_list(sci_table)
        # Filter for desired files
        table = Observations.filter_products(
            data_products_by_id, productSubGroupDescription=subgroup,
            calib_level=calib_level)
    else:
        raise ValueError(f"No data found for proposal {proposal_id}, "
                         f"observation {observation_id}, visit {visit_id}")

    if len(table) == 0:
        raise ValueError(f"No data found for proposal {proposal_id}, "
                         f"observation {observation_id}, visit {visit_id} "
                         f"with subgroup {subgroup} and calib_level "
                         f"{calib_level}")

    # Sort so that the files are downloaded in a sensible order
    if 'dataURI' not in table.colnames:
        raise ValueError("No dataURI column found in the table. "
                         "This probably means no files were found.")
    table.sort('dataURI')

    return table


def download_files(proposal_id, obs_id, visit_id, download_dir,
                   subgroup='UNCAL', flat=True, verbose=False):
    """Download files from MAST based on the provided IDs.

    Parameters:
    -----------
    proposal_id : int
        The JWST proposal ID.
    obs_id : int
        The observation ID.
    visit_id : int
        The visit ID.
    download_dir : str
        The directory where files will be downloaded.
    subgroup : str, optional
        The subgroup of files to download (e.g., 'UNCAL', 'RATEINTS').
    flat : bool, optional
        If True, downloads files in a flat structure (default is True).
    verbose : bool, optional
        If True, prints detailed information about the download process
        (default is False).

    Returns:
    --------
    bool
        True if all files were downloaded successfully, False otherwise.
    """
    if verbose:
        log("[STARTING]", CYAN,
            f"Downloading {subgroup} files for proposal {proposal_id}, "
            f"observation {obs_id}, visit {visit_id}")

    table = query_MAST(proposal_id, obs_id, visit_id, subgroup=subgroup)

    if verbose:
        log("[INFO]", CYAN, f"Scheduling {len(table)} files for download")

    all_success = True
    for product in table:
        filename = os.path.basename(product["productFilename"])

        try:
            manifest = Observations.download_products(
                product, download_dir=download_dir, flat=flat)
        except KeyboardInterrupt:
            raise  # Let Ctrl+C interrupt the program
        except Exception as e:
            log("[ERROR]", RED,
                f"Exception during download of {filename}: {e}")
            all_success = False
            continue

        entry = manifest[0]
        if entry["Status"] != "COMPLETE":
            all_success = False
            log("[FAILED]", RED, f"{filename}")
            log("[ERROR]", YELLOW, f"Status: {entry['Status']}")
            log("[ERROR]", YELLOW, f"Message: {entry['Message']}")

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Download RW-DDT JWST _rateints and _uncal files.")
    parser.add_argument("proposal_id", type=int,
                        help="JWST proposal ID (e.g., 9235)")
    parser.add_argument("observation_ids", type=str,
                        help="Comma-separated observation IDs (e.g., 1 for " +
                        "just the first observation or 4,5 to combine " +
                        "observations 4 and 5)")
    parser.add_argument("visit_id", type=int,
                        help="Visit ID to name output subdirectory (e.g., 1 " +
                        "for the first eclipse observation)")
    parser.add_argument(
        "--base_dir",
        default="/astro/rwddt/JWST/",
        help="Base directory under which target/visit folders will be " +
        "created. Defaults to /astro/rwddt/JWST/"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress download progress messages"
    )
    args = parser.parse_args()

    proposal_id = args.proposal_id
    observation_ids = [int(x.strip()) for x in args.observation_ids.split(",")]
    visit_id = args.visit_id
    base_dir = args.base_dir
    verbose = not args.quiet

    # Download and parse XML
    url = ("https://www.stsci.edu/jwst-program-info/visits/" +
           f"?program={proposal_id}&download=")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        log("[ERROR]", RED, f"Failed to fetch program XML: {e}")
        sys.exit(1)
    root = ET.fromstring(response.content)

    # Use first observation ID to determine target
    first_obs_id = observation_ids[0]
    first_visit = root.find(f".//visit[@observation='{first_obs_id}']")
    if first_visit is None:
        raise ValueError(f"Visit with observation={first_obs_id} not found")

    raw_target = first_visit.findtext("target", default="UnknownTarget")
    target = normalize_target(raw_target)

    output_dir = os.path.join(base_dir, target, f"visit{visit_id}")
    os.makedirs(output_dir, exist_ok=True)

    # The following will make a new directory to store the _uncal files
    uncaldir = os.path.join(output_dir, 'Uncalibrated')
    if not os.path.exists(uncaldir):
        os.makedirs(uncaldir, exist_ok=True)

    # The following will make a new directory to store the _rateints files
    s1dir = os.path.join(output_dir, 'Analysis_A/Quicklook/MAST_Stage1')
    if not os.path.exists(s1dir):
        os.makedirs(s1dir, exist_ok=True)

    # Process each observation
    allSuccess = True
    for obs_id in observation_ids:
        visit = root.find(f".//visit[@observation='{obs_id}']")
        visit_id = int(visit.attrib['visit'])

        if visit is None:
            raise ValueError(f"Visit with observation={obs_id} not found")

        status = visit.findtext("status", default="N/A")

        if status != "Executed":
            raise ValueError(
                f"Observation {obs_id} has status '{status}', "
                "which is not 'Executed'. Cannot download data.")

        log("[INFO]", CYAN,
            f"Will save files from observation {obs_id} into {output_dir}")

        # First get the rateints since they're smaller and faster to download
        allSuccess1 = download_files(proposal_id, obs_id, visit_id, s1dir,
                                     'RATEINTS', verbose=verbose)

        # Next get the uncal files
        allSuccess2 = download_files(proposal_id, obs_id, visit_id, uncaldir,
                                     'UNCAL', verbose=verbose)

        if allSuccess1 and allSuccess2:
            log("[SUCCESS]", GREEN,
                f"All files downloaded for observation {obs_id}")
        else:
            log("[WARNING]", RED,
                f"Some files failed to download for observation {obs_id}. "
                "Check the above outputs for details.")
            allSuccess = False

    if allSuccess:
        log("[DONE]", GREEN, "All downloads complete.")
    else:
        log("[DONE]", RED, "Process completed, but some downloads failed.")


if __name__ == "__main__":
    main()
