import sys
import csv
import os
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QFileDialog

class CsvViewerDialog(QDialog):
    def __init__(self, csv_path):
        super().__init__()
        self.windowCaption = os.path.splitext(os.path.basename(csv_path))[0]
        self.setWindowTitle(self.windowCaption + " Viewer")
        self.resize(800, 600)
        # Layout to hold the table and other widgets
        layout = QVBoxLayout()

        # Table widget to display the CSV content
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        # Load the CSV file and populate the table
        self.load_csv(csv_path)
        
        self.setLayout(layout)

    def load_csv(self, csv_path):
        try:
            with open(csv_path, 'r') as csvfile:
            # with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                
                # Set the number of rows and columns based on the CSV data
                if rows:
                    self.table.setRowCount(len(rows))
                    self.table.setColumnCount(len(rows[0]))
                    

                    # Initialize a list to track the maximum length of content in each column
                    max_column_width = [0] * len(rows[0])

                    # Fill the table with CSV data
                    for row_idx, row in enumerate(rows):
                        for col_idx, cell in enumerate(row):
                            self.table.setItem(row_idx, col_idx, QTableWidgetItem(cell))

                            # Update the maximum width of the column based on the current cell's content
                            max_column_width[col_idx] = max(max_column_width[col_idx], len(cell))
                                # Set the column widths based on the maximum content length

                for col_idx, width in enumerate(max_column_width):
                    # Add some padding to the width to make the table look better
                    self.table.setColumnWidth(col_idx, width * 15)  # Adjust the factor as needed
        except Exception as e:
            print(f"Error loading CSV file: {e}")