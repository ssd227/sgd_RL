import logo from './logo.svg';
import './App.css';
import { useState } from 'react';


function Square({ value, onSquareClick }) {
  return (
    <button className="square" onClick={onSquareClick}>
      {value}
    </button>
  );
}

function Board({ boardSize, xIsNext, squares, onPlay }) {

  function handleClick(i) {
    if (squares[i] || calculateWinner(squares)) {
      return;
    }
    const nextSquares = squares.slice();
    if (xIsNext) {
      nextSquares[i] = "⚫";
    } else {
      nextSquares[i] = "⚪";
    }
    onPlay(nextSquares);
  }

  const winner = calculateWinner(squares);
  let status;
  if (winner) {
    status = "Winner: " + winner;
  } else {
    status = "Next player: " + (xIsNext ? "⚫" : "⚪");
  }



  return (
    <>
      <div className="status">{status}</div>
      {Array(boardSize).fill(0).map((_, rowi) => {
          return(
              <div className="board-row" key={rowi}>
                {Array(boardSize).fill(0).map((_, colj) => {
                    const ij = rowi*boardSize+colj
                    return (
                      <Square value={squares[ij]} onSquareClick={() => handleClick(ij)} key={ij} />);
                  })}
              </div>);})}
    </>
  );
}

export default function Game() {

  const [history, setHistory] = useState([Array(9).fill(null)]); // history设置后仍然会保留旧值
  const [currentMove, setCurrentMove] = useState(0);
  const xIsNext = currentMove % 2 === 0; // 由当前move可以直接推断出 paly-black or play-white

  // 也就是说这种history.length 二阶值得改变，不会对界面上得物件做刷新。很容易出bug
  // const currentSquares = history[history.length - 1]; // 为什么这个更新不会自动刷新
  const currentSquares = history[currentMove]; // 这里之所以会re-render是由于currentMove是状态变量
  const boardSize = 9; 

  function handlePlay(nextSquares) {
    const nextHistory = [...history.slice(0, currentMove + 1), nextSquares];
    setHistory(nextHistory);
    setCurrentMove(nextHistory.length - 1); // 不是废话，通常更新game state时用到
  }

  function jumpTo(nextMove) {
    setCurrentMove(nextMove);
  }

  const moves = history.map((squares, id) => {
    let description;
    if (id > 0) {
      description = 'Go to move #' + id;
    } else {
      description = 'Go to game start';
    }
    return (
      <li key={id}>
        <button onClick={() => jumpTo(id)}>{description}</button>
      </li>
    );
  });

  return (
    <div className="game">
      <div className="game-board">
        <Board boardSize={boardSize} xIsNext={xIsNext} squares={currentSquares} onPlay={handlePlay} />
      </div>
      <div className="game-info">
        <ol>{moves}</ol>
      </div>
    </div>
  );
}

function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
  ];
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}
