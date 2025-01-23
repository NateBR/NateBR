import os
import pickle
import psutil
import sqlite3
import threading
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QCheckBox, QHeaderView
)

# Classe para gerenciamento do banco de dados
class DatabaseManager:
    def __init__(self, db_name="process_data.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_data (
                process_name TEXT PRIMARY KEY,
                usage_count INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def update_process(self, process_name):
        self.cursor.execute("INSERT OR IGNORE INTO process_data (process_name) VALUES (?)", (process_name,))
        self.cursor.execute("UPDATE process_data SET usage_count = usage_count + 1 WHERE process_name = ?", (process_name,))
        self.conn.commit()

    def get_all_data(self):
        self.cursor.execute("SELECT * FROM process_data")
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

# Classe para o Sistema de Machine Learning
class ProcessClassifier:
    MODEL_FILE = "model.pkl"
    VECTORIZER_FILE = "vectorizer.pkl"

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.model = None
        self.vectorizer = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        if os.path.exists(self.MODEL_FILE) and os.path.exists(self.VECTORIZER_FILE):
            with open(self.MODEL_FILE, "rb") as model_file, open(self.VECTORIZER_FILE, "rb") as vectorizer_file:
                self.model = pickle.load(model_file)
                self.vectorizer = pickle.load(vectorizer_file)
        else:
            self.train_model()

    def train_model(self):
        data = self.db_manager.get_all_data()
        if not data:
            return

        process_names = [item[0] for item in data]
        usage_counts = [item[1] for item in data]

        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(process_names)
        y = ["essential" if count > 5 else "terminable" for count in usage_counts]

        self.model = MultinomialNB()
        self.model.fit(X, y)

        # Salvar modelo e vetorizador
        with open(self.MODEL_FILE, "wb") as model_file, open(self.VECTORIZER_FILE, "wb") as vectorizer_file:
            pickle.dump(self.model, model_file)
            pickle.dump(self.vectorizer, vectorizer_file)

    def predict(self, process_name):
        if not self.model or not self.vectorizer:
            return "unknown"
        try:
            X_test = self.vectorizer.transform([process_name])
            return self.model.predict(X_test)[0]
        except Exception:
            return "unknown"

# Função para Encerrar Processos
def terminate_process(process_name, critical_processes):
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] == process_name and process_name not in critical_processes:
                proc.terminate()
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

# Interface Principal
class MonitorApp(QMainWindow):
    CRITICAL_PROCESSES = ["explorer.exe", "System", "svchost.exe"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitor de Otimização do PC")
        self.setGeometry(100, 100, 800, 600)

        # Inicializar banco de dados e modelo
        self.db_manager = DatabaseManager()
        self.classifier = ProcessClassifier(self.db_manager)
        self.automation_enabled = False

        # Layout principal
        layout = QVBoxLayout()
        self.cpu_label = QLabel("Uso de CPU: 0%")
        self.memory_label = QLabel("Uso de Memória: 0%")
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.memory_label)

        # Tabela de processos
        self.process_table = QTableWidget(0, 3)
        self.process_table.setHorizontalHeaderLabels(["Nome do Processo", "Uso de CPU (%)", "Classificação"])
        self.process_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.process_table)

        # Botão de Automação
        self.automation_checkbox = QCheckBox("Ativar Automação")
        self.automation_checkbox.stateChanged.connect(self.toggle_automation)
        layout.addWidget(self.automation_checkbox)

        # Botão de Atualização Manual
        refresh_button = QPushButton("Atualizar Agora")
        refresh_button.clicked.connect(self.update_monitor)
        layout.addWidget(refresh_button)

        # Widget central
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer para Atualização
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_monitor)
        self.timer.start(2000)  # Atualiza a cada 2 segundos

    def update_monitor(self):
        threading.Thread(target=self._collect_and_update).start()

    def _collect_and_update(self):
        cpu_usage = psutil.cpu_percent(interval=0.5)
        memory_usage = psutil.virtual_memory().percent

        # Atualizar a interface
        self.cpu_label.setText(f"Uso de CPU: {cpu_usage}%")
        self.memory_label.setText(f"Uso de Memória: {memory_usage}%")

        # Obter processos ativos
        processes = []
        for proc in psutil.process_iter(['name', 'cpu_percent']):
            try:
                if proc.info['cpu_percent'] > 0:
                    processes.append((proc.info['name'], proc.info['cpu_percent']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Atualizar tabela
        self.process_table.setRowCount(0)
        for process_name, cpu_percent in processes:
            self.db_manager.update_process(process_name)
            classification = self.classifier.predict(process_name)

            row_position = self.process_table.rowCount()
            self.process_table.insertRow(row_position)
            self.process_table.setItem(row_position, 0, QTableWidgetItem(process_name))
            self.process_table.setItem(row_position, 1, QTableWidgetItem(f"{cpu_percent}%"))
            self.process_table.setItem(row_position, 2, QTableWidgetItem(classification))

            # Encerrar processos, se automação estiver habilitada
            if self.automation_enabled and classification == "terminable":
                terminate_process(process_name, self.CRITICAL_PROCESSES)

    def toggle_automation(self, state):
        self.automation_enabled = state == Qt.Checked
