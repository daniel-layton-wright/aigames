from PyQt5.QtWidgets import *
from PyQt5.QtGui     import *
from PyQt5.QtCore    import *
from aigames.game.connect4 import Connect4State, Connect4
from aigames.game.game import GameListener
from aigames.agent import ManualAgent, Agent
from aigames.agent.alpha_agent import AlphaAgent, DummyAlphaEvaluator
import time
import multiprocessing as mp


class Connect4ClickAgent(Agent):
    def __init__(self):
        self.waiting = True
        self.action = None

    def get_action(self, state, legal_actions) -> int:
        self.waiting = True
        action = None
        while action not in legal_actions:
            while self.waiting:
                qApp.processEvents()
                time.sleep(.1)

            action = self.action

        return legal_actions.index(action)


class Connect4Gui(GameListener):
    def __init__(self, agent: Connect4ClickAgent=None, show_neighbor_counts=True):
        self.agent = agent
        self.app = QApplication([])
        self.widget = Connect4Widget(self, Connect4State(), show_neighbor_counts=show_neighbor_counts)

    def before_game_start(self, game):
        self.widget.update_state(game.state)
        self.widget.show()
        qApp.processEvents()

    def after_action(self, game):
        self.widget.update_state(game.state)
        qApp.processEvents()

    def on_mouse_click(self, col_index):
        if self.agent:
            self.agent.waiting = False
            self.agent.action = col_index


class Connect4Widget(QWidget):
    def __init__(self, parent: Connect4Gui, state: Connect4State, show_neighbor_counts=True):
        super().__init__()
        self.parent = parent
        self.state = state
        self.ROWS, self.COLS = self.state.grid[0].numpy().shape
        self.CELL_SIZE = 100
        self.setFixedSize(self.CELL_SIZE*self.COLS, self.CELL_SIZE*self.ROWS)
        self.setWindowTitle('Connect4')
        self.has_been_painted = False
        self.show_neighbor_counts = show_neighbor_counts
        self.process = QProcess(self)
        self.show()

    def update_state(self, state: Connect4State):
        self.state = state
        self.repaint()

    def paintEvent(self, a0: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setPen(Qt.black)
        for i in range(self.COLS + 1):
            painter.drawLine(self.CELL_SIZE*i, 0, self.CELL_SIZE*i, self.ROWS*self.CELL_SIZE)

        for i in range(self.ROWS + 1):
            painter.drawLine(0, self.CELL_SIZE * i, self.COLS * self.CELL_SIZE, self.CELL_SIZE * i)

        for i in range(self.ROWS):
            for j in range(self.COLS):
                if self.state.grid[0, i, j] == 1:  # red
                    self.paintMarker(i, j, painter, Qt.red)
                elif self.state.grid[0, i, j] == -1:  # blue
                    self.paintMarker(i, j, painter, Qt.blue)

    def paintMarker(self, i, j, painter, color):
        self.has_been_painted = True
        x = (j + 0.5) * self.CELL_SIZE
        y = (i + 0.5) * self.CELL_SIZE
        r = self.CELL_SIZE / 3.0
        painter.setPen(QPen(color, 0, Qt.SolidLine))
        painter.setBrush(QBrush(color, Qt.SolidPattern))
        painter.drawEllipse(QPoint(x, y), r, r)

        if self.show_neighbor_counts:
            h = self.CELL_SIZE / 6.0
            w = self.CELL_SIZE / 3.0
            font_size = self.CELL_SIZE/6.0
            _, _, K, L = self.state.neighbors.shape
            for k in range(K):
                for l in range(L):
                    if k == 1 and l == 1:
                        continue
                    left = self.CELL_SIZE * (j + l/2.0) - int(l > 0) * font_size/1.6
                    top = self.CELL_SIZE * (i + k/2.) - int(k > 0) * font_size/1.3
                    painter.setFont(QFont('Terminus', pointSize=font_size))
                    painter.drawText(QRect(left, top , w, h), Qt.AlignLeft, str(int(self.state.neighbors[i, j, k, l])))

    def mouseReleaseEvent(self, a0: QMouseEvent) -> None:
        col_index = (int(a0.localPos().x() / self.CELL_SIZE))
        self.parent.on_mouse_click(col_index)


def main():
    manual_agent = Connect4ClickAgent()
    m = ManualAgent()
    gui = Connect4Gui(manual_agent, show_neighbor_counts=True)
    evaluator = DummyAlphaEvaluator(len(Connect4.get_all_actions()))
    agent = AlphaAgent(Connect4, evaluator, [], use_tqdm=True)
    game = Connect4([agent, manual_agent], [gui])
    game.play()

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    mp.set_start_method("spawn")
    main()
