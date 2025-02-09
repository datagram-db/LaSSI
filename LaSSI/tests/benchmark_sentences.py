from LaSSI.tests.delete_catabolites import delete_files
from LaSSI.tests.run_all_sentences import get_and_run_all_sentences


def benchmark_sentences():
    for j in range(num_of_iterations):
        get_and_run_all_sentences(["benchmarking"])
        delete_files()


if __name__ == '__main__':
    num_of_iterations = 5
    should_rerun_meuDB_generation = True

    if should_rerun_meuDB_generation:
        for i in range(num_of_iterations):
            delete_files(True, True)  # First delete ALL files (inc. meuDB) and only benchmarking files
            benchmark_sentences()
    else:
        benchmark_sentences()
