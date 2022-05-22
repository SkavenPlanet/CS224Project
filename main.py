import os
import sys
import chess
import chess.engine
import pyttsx3

from PyQt5.QtCore import *
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import *

from nemo.collections.nlp.models import IntentSlotClassificationModel
from nemo.collections import nlp as nemo_nlp
from nemo.utils.exp_manager import exp_manager
from nemo.utils import logging

import os
import wget
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

pieces = {'p' : 'pawn', 'b' : 'bishop', 'k' : 'king', 'q' : 'queen', 'r' : 'rook', 'n' : 'knight'}

#dictate_move slots
# piece
# start_square
# end_square

# def validate_square (txt):
#     type(txt) == type()
#     len(txt) == 2

class MainWindow(QWidget):
    """
    Create a surface for the chessboard.
    """
    def __init__(self):
        """
        Initialize the chessboard.
        """
        super().__init__()

        self.setWindowTitle("Chess GUI")
        self.setGeometry(300, 300, 1400, 800)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 600, 600)

        self.boardSize = min(self.widgetSvg.width(),
                             self.widgetSvg.height())

        self.text_area = QTextEdit()
        self.text_area.setFocusPolicy(Qt.NoFocus)
        message = QLineEdit()

        def user_submit():
            message.text()
            txt = message.txt()
            self.process_input(txt)
            self.text_area.append("USER: " + message.text())
            message.clear()

        message.returnPressed.connect(user_submit)

        layout = QGridLayout()
        layout.addWidget(self.text_area, 0, 0)
        layout.addWidget(message, 1, 0)
        layout.addWidget(self.widgetSvg, 0, 1)
        self.setLayout(layout)

        self.coordinates = True
        self.margin = 0.05 * self.boardSize if self.coordinates else 0
        self.squareSize = (self.boardSize - 2 * self.margin) / 8.0
        self.pieceToMove = [None, None]

        self.board = chess.Board()
        self.drawBoard()
        self.engine = chess.engine.SimpleEngine.popen_uci("engine/stockfish_15_x64_avx2.exe")
        self.speech_engine = pyttsx3.init()

    def process_input (self, txt):
        return

    def say_response (self, txt):
        self.text_area.append("CPU: " + txt)
        self.speech_engine.say(txt)

    @pyqtSlot(QWidget)
    def mousePressEvent(self, event):
        """
        Handle left mouse clicks and enable moving chess pieces by
        clicking on a chess piece and then the target square.

        Moves must be made according to the rules of chess because
        illegal moves are suppressed.
        """
        if event.x() <= self.boardSize and event.y() <= self.boardSize:
            if event.buttons() == Qt.LeftButton:
                if self.margin < event.x() < self.boardSize - self.margin and self.margin < event.y() < self.boardSize - self.margin:
                    file = int((event.x() - self.margin) / self.squareSize)
                    rank = 7 - int((event.y() - self.margin) / self.squareSize)
                    square = chess.square(file, rank)
                    piece = self.board.piece_at(square)
                    coordinates = "{}{}".format(chr(file + 97), str(rank + 1))
                    if self.pieceToMove[0] is not None:
                        move = chess.Move.from_uci("{}{}".format(self.pieceToMove[1], coordinates))
                        if move in self.board.legal_moves:
                            before = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))['score'].white()
                            self.board.push(move)
                            after = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))['score'].white()
                            print ("Before: ", before, " After: ", after)
                            good_move = after > before
                            self.say_response("good move" if good_move else "bad move")
                            ai_move = self.engine.play(self.board, chess.engine.Limit(time=0.01))
                            print(ai_move.move)
                            move_str = str(ai_move.move)
                            piece = pieces[str(self.board.piece_at(chess.parse_square(move_str[0:2])))]
                            print(piece)
                            self.board.push(ai_move.move)
                            self.say_response("{} {} to {}".format(piece, move_str[:2], move_str[2:]))
                            self.speech_engine.runAndWait()
                        piece = None
                        coordinates = None
                    self.pieceToMove = [piece, coordinates]
                    self.drawBoard()

    def drawBoard(self):
        """
        Draw a chessboard with the starting position and then redraw
        it for every new move.
        """
        self.boardSvg = self.board._repr_svg_().encode("UTF-8")
        self.drawBoardSvg = self.widgetSvg.load(self.boardSvg)
        self.widgetSvg.setFixedWidth(600)

        return self.drawBoardSvg


if __name__ == "__main__":
    config_file = "intent_slot_classification_config.yaml"
    config = OmegaConf.load(config_file)

    nlu_model = IntentSlotClassificationModel.restore_from(restore_path="nlumodel.nemo")

    queries = [
        'set alarm for seven thirty am',
        'lower volume by fifty percent',
        'what is my schedule for tomorrow',
    ]

    pred_intents, pred_slots = nlu_model.predict_from_examples(queries, config.model.test_ds)

    logging.info('The prediction results of some sample queries with the trained model:')
    for query, intent, slots in zip(queries, pred_intents, pred_slots):
        logging.info(f'Query : {query}')
        logging.info(f'Predicted Intent: {intent}')
        logging.info(f'Predicted Slots: {slots}')

    chessGui = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(chessGui.exec_())