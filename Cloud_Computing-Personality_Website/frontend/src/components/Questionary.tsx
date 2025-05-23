import React, { useState } from 'react';
import axios from 'axios';
import { Pie, Bar } from 'react-chartjs-2';
import { questions, options, open_questions, open_question_answers } from './questionsAndOptions';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement
} from 'chart.js';

// Register the necessary chart components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const Questionary: React.FC = () => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://flaskapp-service.my-app.svc.cluster.local:4000';

  const [answers, setAnswers] = useState<(number | null)[]>(Array(questions.length).fill(null));
  const [openAnswers, setOpenAnswers] = useState<string[]>(Array(open_questions.length).fill(''));
  const [error, setError] = useState<string | null>(null);
  const [generatedText, setGeneratedText] = useState<string | null>(null);
  const [bigFiveResult, setBigFiveResult] = useState<any>(null);
  const [mbtiResult, setMbtiResult] = useState<any>(null);
  const [careerDataResult, setCareerDataResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleAnswerChange = (questionIndex: number, optionIndex: number) => {
    const newAnswers = [...answers];
    newAnswers[questionIndex] = optionIndex;
    setAnswers(newAnswers);
  };

  const handleOpenAnswerChange = (questionIndex: number, value: string) => {
    const newOpenAnswers = [...openAnswers];
    newOpenAnswers[questionIndex] = value;
    setOpenAnswers(newOpenAnswers);
  };

  const generateRandomAnswers = () => {
    const randomAnswers = questions.map(() => Math.floor(Math.random() * options.length));
    setAnswers(randomAnswers);

    const randomOpenAnswers = open_questions.map((_, index) => {
      const key = `answers_open_question_${index + 1}`;
      const answersArray = open_question_answers[key as keyof typeof open_question_answers];
      const randomIndex = Math.floor(Math.random() * answersArray.length);
      return answersArray[randomIndex];
    });
    setOpenAnswers(randomOpenAnswers);
  };

  const generateText = () => {
    const questionText = questions.map((question, index) => {
      const answerIndex = answers[index];
      if (answerIndex !== null) {
        if (index < 40) {
          return `This is ${options[answerIndex]} that ${question}`;
        } else {
          return `This is ${options[answerIndex]} that I am ${question}`;
        }
      }
      return '';
    }).join(' ');

    const openQuestionText = open_questions.map((question, index) => {
      return `For the question "${question}", my answer is "${openAnswers[index]}".`;
    }).join(' ');

    return `${questionText} ${openQuestionText}`;
  };

  const getRandomColor = () => {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  };

  const handleSubmit = async () => {
    const allAnswered = answers.every(answer => answer !== null);
    const allOpenAnswered = openAnswers.every(answer => answer.trim() !== '');
    if (!allAnswered || !allOpenAnswered) {
      setError("Please answer all the questions before submitting.");
      return;
    }
    setError(null);
    setGeneratedText(null);
    setBigFiveResult(null);
    setMbtiResult(null);
    setCareerDataResult(null);
    setLoading(true);

    const text = generateText();
    setGeneratedText(text);

    try {
      const bigFiveResponse = await axios.post(`${apiUrl}/classify_big_five`, { text });
      setBigFiveResult(bigFiveResponse.data);
      
      const mbtiResponse = await axios.post(`${apiUrl}/classify_mbti`, { text });
      setMbtiResult(mbtiResponse.data);

      const careerDataResponse = await axios.post(`${apiUrl}/get_career_recommendations`, { text });
      setCareerDataResult(careerDataResponse.data);
      
      setLoading(false);

    } catch (error) {
      console.error("Error classifying text:", error);
      setError("There was an error processing your request. Please try again.");
      setLoading(false);
    }
  };

  const preparePieChartData = (data: any) => {
    return {
      labels: Object.keys(data),
      datasets: [
        {
          data: Object.values(data),
          backgroundColor: Object.keys(data).map(() => getRandomColor()),
        }
      ]
    };
  };

  const prepareBarChartData = (data: any) => {
    return {
      labels: Object.keys(data),
      datasets: [
        {
          label: 'Big Five Personality Traits',
          data: Object.values(data),
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1,
        },
      ],
    };
  };

  return (
    <div className="p-4 flex justify-center">
      <div className="w-4/5">
        <h1 className="text-2xl font-bold mb-6 text-center">Personality Questionnaire</h1>
        <p className="mt-4 mb-6 text-2xl font-medium text-left">Rate each statement according to how well it describes you. Base your ratings on how you really are, not how you would like to be.</p>
        {questions.map((question, qIndex) => (
          <div key={qIndex} className="mb-8">
            <p className="mb-2 text-lg text-left">{qIndex + 1}. {question}</p>
            <div className="flex items-center justify-center space-x-2">
              <div className="flex justify-between flex-1 space-x-2">
                {options.map((option, oIndex) => (
                  <label key={oIndex} className="flex-1 text-center">
                    <input
                      type="radio"
                      name={`question-${qIndex}`}
                      value={oIndex}
                      checked={answers[qIndex] === oIndex}
                      onChange={() => handleAnswerChange(qIndex, oIndex)}
                      className="hidden"
                    />
                    <div className={`p-2 border-2 ${answers[qIndex] === oIndex ? 'border-blue-500' : 'border-gray-300'} rounded-md cursor-pointer`}>
                      {oIndex + 1}
                    </div>
                  </label>
                ))}
              </div>
            </div>
            {qIndex === 39 && (
              <p className="mt-4 mb-6 text-2xl font-medium text-left">
                Rate each word according to how well it describes you. Base your ratings on how you really are, not how you would like to be.
              </p>
            )}
          </div>
        ))}
        <p className="mt-4 mb-6 text-2xl font-medium text-left">Open Questions</p>
        {open_questions.map((question, oIndex) => (
          <div key={oIndex} className="mb-8">
            <p className="mb-2 text-lg font-medium text-left">{oIndex + 1}. {question}</p>
            <div className="flex items-center justify-center">
              <textarea
                value={openAnswers[oIndex]}
                onChange={(e) => handleOpenAnswerChange(oIndex, e.target.value)}
                className="w-full p-2 border-2 border-gray-300 rounded-md"
                style={{ height: '100px' }} // Increase the height of the textarea
              />
            </div>
          </div>
        ))}
        {error && <p className="text-red-500 mb-4 text-center">{error}</p>}
        <div className="flex justify-center space-x-4">
          <button
            onClick={handleSubmit}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Submit
          </button>
          <button
            onClick={generateRandomAnswers}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          >
            Randomize Answers
          </button>
        </div>
        {loading && (
          <div className="mt-6 p-4 bg-gray-100 rounded text-center">
            <h2 className="text-xl font-bold mb-4">Loading...</h2>
            <div className="spinner-border animate-spin inline-block w-8 h-8 border-4 rounded-full" role="status">
              <span className="visually-hidden">X</span>
            </div>
          </div>
        )}
        {!loading && generatedText && bigFiveResult && mbtiResult && careerDataResult && (
          <>
            <div className="mt-6 p-4 bg-gray-100 rounded text-center">
              <h2 className="text-xl font-bold mb-4">Generated Text</h2>
              <p>{generatedText}</p>
            </div>
            <div className="flex flex-wrap justify-center mt-6">
              <div className="chart-container w-full md:w-1/2 p-4">
                <h2 className="text-xl font-bold mb-4 text-center">Big Five Classification Results</h2>
                <Bar data={prepareBarChartData(bigFiveResult)} />
              </div>
              <div className="chart-container w-full md:w-1/2 p-4">
                <h2 className="text-xl font-bold mb-4 text-center">MBTI Classification Results</h2>
                <Pie data={preparePieChartData(mbtiResult)} />
              </div>
            </div>
            <h2 className="text-xl font-bold mb-4 text-center">Career Suggestions Sorted by Salary in US</h2>
            <div className="flex flex-wrap justify-center mt-6">
              {careerDataResult.MBTI.map((mbti: any, index: number) => {
                const type = Object.keys(mbti)[0];
                const careers = mbti[type];
                return (
                  <div className="w-full md:w-1/2 p-4" key={`mbti-${index}`}>
                    <h3 className="text-lg font-bold mb-4 text-center">{type}</h3>
                    <table className="min-w-full bg-white text-left">
                      <thead>
                        <tr>
                          <th className="py-2 px-4">#</th>
                          <th className="py-2 px-4">Career</th>
                          <th className="py-2 px-4">Average Salary (USD)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {careers.map((career: any, careerIndex: number) => (
                          <tr key={careerIndex}>
                            <td className="border px-4 py-2">{careerIndex + 1}</td>
                            <td className="border px-4 py-2">{career.Career}</td>
                            <td className="border px-4 py-2">{career["Average Salary (USD)"]}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                );
              })}
              {careerDataResult.big_5.map((big5: any, index: number) => {
                const trait = Object.keys(big5)[0];
                const careers = big5[trait];
                return (
                  <div className="w-full md:w-1/2 p-4" key={`big5-${index}`}>
                    <h3 className="text-lg font-bold mb-4 text-center">{trait}</h3>
                    <table className="min-w-full bg-white text-left">
                      <thead>
                        <tr>
                          <th className="py-2 px-4">#</th>
                          <th className="py-2 px-4">Career</th>
                          <th className="py-2 px-4">Average Salary (USD)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {careers.map((career: any, careerIndex: number) => (
                          <tr key={careerIndex}>
                            <td className="border px-4 py-2">{careerIndex + 1}</td>
                            <td className="border px-4 py-2">{career.Career}</td>
                            <td className="border px-4 py-2">{career["Average Salary (USD)"]}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Questionary;
