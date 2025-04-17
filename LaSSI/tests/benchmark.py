import csv
from typing import Dict, List

class Benchmark:
    _data: List[Dict[str, float]] = []
    _phase_names: List[str] = []

    def __init__(self):
        self.data = Benchmark._data
        self.phase_names = Benchmark._phase_names

    def add_row(self, sentence_id: int, phase_name: str, time: float) -> None:
        if phase_name not in self.phase_names:
            self.phase_names.append(phase_name)

        for row in self.data:
            if row['sentence_id'] == sentence_id:
                row[phase_name] = time
                return

        new_row = {'sentence_id': sentence_id, phase_name: time}
        if self.data:
            previous_row = self.data[-1]
            for existing_phase in self.phase_names:
                if existing_phase not in new_row:
                    new_row[existing_phase] = 0.0
            for phase in new_row:
                if phase != 'sentence_id' and phase not in previous_row:
                    previous_row[phase] = 0.0
        self.data.append(new_row)

    def to_csv(self, filename: str = 'benchmark_results') -> None:
        if not self.data:
            print("No data to export.")
            return

        filename = f"{filename}_{len(self.data)}.csv"

        try:
            with open(filename, 'w', newline='') as csvfile:
                header = ['sentence_id'] + self.phase_names
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                for row in self.data:
                    full_row = {'sentence_id': row['sentence_id']}
                    for phase in self.phase_names:
                        full_row[phase] = row.get(phase, 0.0)
                    writer.writerow(full_row)
            print(f"Data successfully written to {filename}")
        except Exception as e:
            print(f"An error occurred while writing to CSV: {e}")