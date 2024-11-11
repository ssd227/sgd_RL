import './App.css';
import { useState} from 'react';
import { useEffect } from 'react';

const val2icon = {
  0 : null, // 不显示
  1 : '⚫',
  2 : '⚪',
};

function Square({ value, onSquareClick }) {
  return (
    <button className="square" onClick={onSquareClick}>
      {value}
    </button>
  );
}

function Board({board, winner, onClick}) {

  let status;
  if (winner) {
    status = "Winner: " + val2icon[winner];
  } else {
    status = "your turn: " + "⚫"; // todo目前玩家总是黑色
  }

  return (
    <>
      <div className="status">{status}</div>
      {board.map((row, i) =>
        <div className="board-row" key={i}>
          {row.map((cell, j) => 
            <Square value={val2icon[cell]} onSquareClick={() => onClick(i, j)} key={i+'-'+j} />)}
        </div>)}
    </>
  );
}

export default function Game() {
  const boardSize = 7; 

  const [board, setBoard] = useState([]);
  const [nextPlayer, setNextPlayer] = useState(1);

  const [isOver, setIsOver] = useState(false);
  const [winner, setWinner] = useState(null);

  const [gameId, setGameId] = useState(10000);

  // 获取初始游戏状态
  useEffect(() => {
    fetch('http://localhost:5000/newgame/'+boardSize)
    .then(response => response.json())
    .then(data => {
      setBoard(data.board); // NxN的一维棋盘, 0表示空位, 1表示黑，2表示白
      setNextPlayer(data.nextPlayer);
      setIsOver(data.isOver);
      setWinner(data.winner);
    });
  }, [gameId]);

  // 处理棋盘点击事件
  const handleClick = (x, y) => {
    if (board[x][y] || isOver) {
      return;
    }

    // todo 发送请求前
    // 先更新当前状态
    // 切换 黑白棋角色UI
    // 等待bot完成请求回复
    

    fetch('http://localhost:5000/make_move', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        x: x,
        y: y,
        player: nextPlayer
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.status === "success") {
        setBoard(data.game_state.board);
        setNextPlayer(data.game_state.current_turn);
        setWinner(data.game_state.winner);
        setIsOver(data.game_state.isOver);
        setWinner(data.game_state.winner);
      } else {
        alert(data.message);  // 处理无效操作
      }
    }); // 通讯结束

  };

  function reStart() {
    // 告诉后端重开游戏
    setGameId(prevId => prevId + 1);
  }

  return (
    <div className="game">
      <div className="game-board">
        <Board board={board} winner={winner} onClick={handleClick} />
      </div>
      <div className="game-info">
        <p> gameID: {gameId}</p>
        <button onClick={() => reStart()}>{'重新开始'}</button>
      </div>
    </div>
  );
}
