import argparse
import json
import logging
import os
import re
import subprocess
import zipfile
from pathlib import Path

import pandas as pd


# pd.set_option('display.max_colwidth', None)  # Allows unlimited column width
# pd.set_option('display.width', 2000)  # Increases the total display width
# pd.set_option('display.max_rows', None)

# Global logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler()  # Stream handler logs to console
#     ]
# )
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("execution.log"), logging.StreamHandler()],
)


def execute_command(command):
    """
    Executes a single command with configurable log file output and captures command output.

    Args:
        command (list): The full command as a list of strings.
        log_file (str): The path to the log file (default is 'execution.log').
    """

    command_str = " ".join(command)
    logging.info(f"Executing command: {command_str}")

    try:
        # Run the command and capture both stdout and stderr
        subprocess.run(command, check=True)
        logging.info(f"Successfully executed: {command_str}")
        return 0

    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while executing command: {command_str}\n{e.stderr}")
        return 1
    except ValueError as e:
        logging.error(f"ValueError occurred: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
        return 1


def download_data_generic(config_paths, timeframes, timerange="20180101-20250130"):
    """
    Prepare the parameter combinations for the 'freqtrade download-data' command.
    """
    logging.info("Starting data download...")
    base_command = ["freqtrade", "download-data", "--prepend"]
    params_list = [
        base_command + ["--config", config, "--timeframe", timeframe, "--timerange", timerange]
        for config in config_paths
        for timeframe in timeframes
    ]
    for command in params_list:
        execute_command(command)
    logging.info("Data download completed.")


def get_strategy_names(strategy_folder):
    """
    Recursively extract the class names that inherit from IStrategy in Python files
    from the specified folder. If a match is found, rename the file to match the class name.
    """
    logging.info(f"Retrieving strategies from folder: {strategy_folder}")
    strategy_names = []
    files_with_proper_strategies = []
    pattern = re.compile(r"^\s*class\s+(\w+)\s*\(\s*IStrategy\s*\)", re.MULTILINE)
    try:
        for root, _, files in os.walk(strategy_folder):
            for i, file in enumerate(files):
                # if i == 0:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        matches = pattern.findall(content)
                        if matches:
                            class_name = matches[0]
                            strategy_names.append(class_name)
                            files_with_proper_strategies.append(file)

                            # Rename the file if the name does not match the class name
                            expected_file_name = f"{class_name}.py"
                            # logging.info(f"expected_file_name: {expected_file_name}, file: {file}")
                            if file != expected_file_name:
                                new_file_path = os.path.join(root, expected_file_name)
                                os.rename(file_path, new_file_path)
                                logging.info(f"Renamed {file} to {expected_file_name}")
                            else:
                                pass
                        else:
                            logging.warning(f"No strategy class found in file: {file}")
                else:
                    logging.warning(f"File {file} is not a Python")

        logging.info(f"Found {len(strategy_names)} strategy classes.")

    except Exception as e:
        logging.error(f"Error retrieving strategy names: {e}")
    return strategy_names, files_with_proper_strategies


def get_timerange(option):
    """
    Returns a timerange based on the selected option.
    """
    timeranges = {
        "all": "20180101-20250130",
        "uptrend": "20241101-20241230",
        "sideway": "20221201-20240930",
        "downtrend": "20210801-20230930",
    }
    selected_timerange = timeranges.get(option, "20180101-20250130")
    logging.info(f"Selected timerange ({option}): {selected_timerange}")
    return selected_timerange


def get_subfolders(path_dir):
    subfolders = [
        str(path_dir / subfolder)
        for subfolder in os.listdir(path_dir)
        if (path_dir / subfolder).is_dir()
    ]
    return subfolders


def get_strategy_batches(strategies, batch_size=100):
    return [strategies[i : i + batch_size] for i in range(0, len(strategies), batch_size)]


def prepare_backtest_command(
    backtest_export_result_path,
    config_paths,
    timeframes,
    timerange,
    strategy_batches,
    strategy_path,
    logfile,
    batch=True,
):
    base_command = [
        "freqtrade",
        "backtesting",
        "--export",
        "trades",
        "--export-filename",
        backtest_export_result_path,
        "--cache",
        "none",
        "--strategy-path",
        strategy_path,
        "--recursive-strategy-search",
    ]
    if batch:
        temp = [
            base_command
            + [
                "--config",
                config_paths,
                "--timeframe",
                timeframes,
                "--timeframe-detail",
                "5m",
                "--timerange",
                timerange,
                "--logfile",
                f"user_data/logs/{logfile}_{i}.log",
                "--strategy-list",
                *batch,
            ]
            for i, batch in enumerate(strategy_batches)
        ]
        return temp
    else:
        temp = [
            base_command
            + [
                "--config",
                config_paths,
                "--timeframe",
                timeframes,
                "--timeframe-detail",
                "5m",
                "--timerange",
                timerange,
                "--logfile",
                f"user_data/logs/{logfile}_{i}.log",
                "--strategy",
                strat,
            ]
            for i, strat in enumerate(strategy_batches)
        ]
        return temp


def move_failed_strategies(
    failed_batches2,
    files_with_proper_strategies,
    repo_path,
    folder_name="strat_ninja_scraped_strategies_manual_process",
):
    for name in failed_batches2:
        if f"{name}.py" in files_with_proper_strategies:
            file_path = os.path.join(repo_path, f"{name}.py")
            new_folder = f"user_data/strategy/{folder_name}"
            os.makedirs(new_folder, exist_ok=True)
            new_file_path = os.path.join(new_folder, f"{name}.py")
            os.rename(file_path, new_file_path)
            logging.info(f"Moved {file_path} to {new_file_path}")


def check_strat_fail_backtest(
    config_paths, timeframes, timerange_option, repo_path, repo_name, batch_size=5
):
    """
    Prepare the parameter combinations for the 'freqtrade backtesting' command.
    """
    logging.info("Starting backtest execution...")
    timerange = get_timerange(timerange_option)

    prefix_backtest_files = f"{repo_name}_{timerange_option}_{timeframes}"
    # backtest_export_result_path = f"user_data/backtest_results/{prefix_backtest_files}.json"

    strategies, files_with_proper_strategies = get_strategy_names(repo_path) if repo_path else []

    # first run to get the initial not failed batch strategy
    logging.info("Running backtest on the initial strategies...")
    # batch_size = 5 #25  # Adjust batch size as needed
    strategy_batches = [
        strategies[i : i + batch_size] for i in range(0, len(strategies), batch_size)
    ]

    logfile = f"{prefix_backtest_files}_checkfail_step1"

    params_list_initial = prepare_backtest_command(
        backtest_export_result_path="",
        config_paths=config_paths,
        timeframes=timeframes,
        timerange=timerange,
        strategy_batches=strategy_batches,
        strategy_path=repo_path,
        logfile=logfile,
        batch=True,
    )

    failed_batches_initial = [
        i for i, command in enumerate(params_list_initial) if execute_command(command) != 0
    ]
    failed_strategies_potential = [
        k
        for fb in failed_batches_initial
        for k in params_list_initial[fb][-batch_size:]
        if k not in ["--strategy-list"]
    ]

    logging.info(f"failed_strategies_potential: {failed_strategies_potential}")

    # Only run finer step if there are failed strategies in the initial step
    if failed_strategies_potential:
        # run finer to get exactly where failed, run each strategy, not batch anymore
        logging.info("Running backtest on the finer failed strategies...")
        logfile = f"{prefix_backtest_files}_checkfail_step2"
        params_list_finer = prepare_backtest_command(
            backtest_export_result_path="",
            config_paths=config_paths,
            timeframes=timeframes,
            timerange=timerange,
            strategy_batches=failed_strategies_potential,
            strategy_path=repo_path,
            logfile=logfile,
            batch=False,
        )

        logging.info(f"params_list_finer: {params_list_finer}")

        failed_strategies_finer = [
            params_list_finer[i][-1]
            for i, command in enumerate(params_list_finer)
            if execute_command(command) != 0
        ]
        logging.warning(f"failed_batches_final: {failed_strategies_finer}")
        # move_failed_strategies(failed_strategies_finer, files_with_proper_strategies, repo_path)
        strategies_remain = [
            strat for strat in strategies if strat not in failed_strategies_finer
        ]  # remove failed_strategies_finer from strategies
        # assert len(strategies_remain) == len(strategies) - len(
        #     failed_strategies_finer
        # ), f"The number of strategies {strategies}, and strategies_remain {strategies_remain} should be reduced by the number of failed strategies."
    else:
        failed_strategies_finer = []
        strategies_remain = strategies

    return strategies, failed_strategies_potential, failed_strategies_finer, strategies_remain


def process_backtest_folder(folder_path, prefix_backtest_files, timerange_option, timeframes):
    # Initialize an empty list to store all strategy comparisons
    strategy_comparisons = []
    extracted_files = []

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".zip") and filename.startswith(f"{prefix_backtest_files}"):
            zip_file_path = os.path.join(folder_path, filename)
            # Unzip the file
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                extracted_files.extend(zip_ref.namelist())
                zip_ref.extractall(folder_path)

            # Look for the .meta.json file
            meta_file_name = filename.replace(".zip", ".json")
            meta_file_path = os.path.join(folder_path, meta_file_name)

            if os.path.exists(meta_file_path):
                # Open and read the JSON content
                with open(meta_file_path, "r") as f:
                    meta_data = json.load(f)

                    # Extract strategy_comparison data
                    if "strategy_comparison" in meta_data:
                        strategy_comparisons.extend(meta_data["strategy_comparison"])

    # Convert the list of strategy comparisons into a DataFrame
    df = pd.DataFrame(strategy_comparisons)

    # Add new columns for timerange_option and timeframes
    df["timerange_option"] = timerange_option
    df["timeframes"] = timeframes

    # Sort by 'profit_mean_pct' in descending order (best profit on top)
    # df_sorted = df.sort_values(by="profit_mean_pct", ascending=False)

    # save df_sorted to the same folder
    df.to_csv(os.path.join(folder_path, f"summary_{prefix_backtest_files}.csv"), index=False)

    # given a folder, look for files with patterns f"{prefix_backtest_files}_something
    for file in extracted_files:
        file_path = os.path.join(folder_path, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")
        else:
            logging.warning(f"File not found: {file_path}")

    return df


def concatenate_summary_df(
    folder_path, repo_name, column_drop, save=True, order_by=["key"]
):
    # Initialize an empty list to store all strategy comparisons
    summary_files = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".csv")
        and filename.startswith(f"summary_{repo_name}")
        and filename != f"summary_{repo_name}_concatenated.csv"
    ]

    df_list = [
        pd.read_csv(file) for file in summary_files
    ]

    df_concatenated = pd.concat(df_list, ignore_index=True)

    # rename the column key by strategy
    df_concatenated.rename(columns={"key": "strategy"}, inplace=True)

    # add column selected with value true or false
    df_concatenated["selected"] = False

    # repositioning columns: [strategy, timerange_option, timeframes, profit_total_pct, profit_mean_pct, profit_total_abs, and remaining columns]
    columns_order = [
        "strategy",
        "timerange_option",
        "timeframes",
        "selected",
        "profit_total_pct",
        "profit_mean_pct",
        "profit_total_abs",
    ] + [col for col in df_concatenated.columns if col not in [
        "strategy",
        "timerange_option",
        "timeframes",
        "selected",
        "profit_total_pct",
        "profit_mean_pct",
        "profit_total_abs",
    ]]
    df_concatenated = df_concatenated[columns_order]

    # Sort by 'profit_mean_pct' in descending order (best profit on top)
    df_sorted = df_concatenated.sort_values(by=order_by, ascending=False)

    # for the column float value, keep one digit only
    df_sorted = df_sorted.round(2)

    # convert the column profit_mean_pct from float to percentage
    df_sorted["max_drawdown_account"] = df_sorted["max_drawdown_account"].apply(
        lambda x: f"{x:.1%}"
    )

    df_sorted["profit_mean_pct"] = df_sorted["profit_mean_pct"].apply(lambda x: f"{x:.1f}%")
    df_sorted["profit_sum_pct"] = df_sorted["profit_sum_pct"].apply(lambda x: f"{x:.1f}%")
    df_sorted["profit_total_pct"] = df_sorted["profit_total_pct"].apply(lambda x: f"{x:.1f}%")

    # drop columns that are not needed
    df_sorted.drop(columns=column_drop, inplace=True)

    assert len(df_sorted) == sum([len(df) for df in df_list]), "The number of rows of concatenated df must be the sum of sub df."

    # save df_sorted to the same folder
    if save:
        df_sorted.to_csv(
            os.path.join(folder_path, f"summary_{repo_name}_concatenated.csv"),
            index=False,
        )

    return df_sorted


def main():
    parser = argparse.ArgumentParser(description="Run backtesting with customizable parameters.")
    parser.add_argument(
        "--timeframes", type=str, default="15m", help="Timeframes for backtest (e.g., '5m', '15m')"
    )
    parser.add_argument(
        "--timerange_option",
        type=str,
        default="sideway",
        choices=["all", "uptrend", "sideway", "downtrend"],
        help="Market condition timerange",
    )
    parser.add_argument(
        "--check_strat_fail_backtest",
        action="store_true",
        help="Whether to run check_strat_fail_backtest before backtest",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for backtesting strategies"
    )
    parser.add_argument(
        "--batch_size_check_fail_backtest",
        type=int,
        default=5,
        help="Batch size for check_strat_fail_backtest",
    )
    parser.add_argument(
        "--concatenate_summary",
        action="store_true",
        help="Whether to run concatenate_summary_df and skip other steps",
    )

    args = parser.parse_args()

    project_root = "/workspaces/freqtrade"
    config_paths = "user_data/config_backtest.json"
    timerange = get_timerange(args.timerange_option)

    strategies_folder = Path(project_root) / "user_data/strategy/strat_ninja_scraped_strategies"
    repos = get_subfolders(strategies_folder)
    if not repos:
        repos = [str(strategies_folder)]

    repo_path = repos[0]
    repo_name = repo_path.split("/")[-1]
    prefix_backtest_files = f"{repo_name}_{args.timerange_option}_{args.timeframes}"
    backtest_export_result_path = f"user_data/backtest_results/{prefix_backtest_files}.json"
    backtest_folder_path = str(Path(backtest_export_result_path).parent)
    logfile = f"{prefix_backtest_files}"

    if args.concatenate_summary:
        logging.info("Concatenating summary files...")
        column_drop = ["profit_mean", "profit_sum", "profit_total", "profit_sum_pct"]
        concatenate_summary_df(
            backtest_folder_path,
            repo_name,
            column_drop,
            save=True,
            order_by=["strategy", "timerange_option", "timeframes"],
        )
        logging.info("Summary concatenation completed successfully.")
        return

    if args.check_strat_fail_backtest:
        logging.info("Running strategy failure check...")
        _, _, _, strategies = check_strat_fail_backtest(
            config_paths=config_paths,
            timeframes=args.timeframes,
            timerange_option=args.timerange_option,
            repo_path=repo_path,
            repo_name=repo_name,
            batch_size=args.batch_size_check_fail_backtest,
        )
    else:
        strategies, files_with_proper_strategies = get_strategy_names(repo_path)

    logging.info("Starting full backtest run...")
    strategy_batches = [
        strategies[i : i + args.batch_size] for i in range(0, len(strategies), args.batch_size)
    ]

    params_list_final = prepare_backtest_command(
        backtest_export_result_path,
        config_paths,
        args.timeframes,
        timerange,
        strategy_batches=strategy_batches,
        strategy_path=repo_path,
        logfile=logfile,
        batch=True,
    )

    for command in params_list_final:
        execute_command(command)

    logging.info("Processing backtest results...")
    process_backtest_folder(
        backtest_folder_path, prefix_backtest_files, args.timerange_option, args.timeframes
    )

    logging.info("All processes completed successfully.")


if __name__ == "__main__":
    main()

# python backtest.py --timeframes 1h --timerange_option downtrend --batch_size 100 --check_strat_fail_backtest --batch_size_check_fail_backtest 5
# python backtest.py --concatenate_summary
