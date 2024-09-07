// src/App.js
import React, { useEffect } from 'react';
import { Route, Routes, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import HomePage from './HomePage';
import SecondPage from './SecondPage';
import SelectPage from './SelectPage';
import PlantDetails from './PlantDetails';
import BudAnalysis from './BudAnalysis';
import BranchAnalysis from './BranchAnalysis'

function App() {
  const location = useLocation();

  useEffect(() => {
    console.log('App component mounted');
  }, []);

  return (
    <div>
      {/* <h1>App Component</h1> */}
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<SelectPage />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/second" element={<SecondPage />} />
          <Route path="/plant-details/:plantId" element={<PlantDetails />} />
          <Route path="/bud-analysis/:plantId" element={<BudAnalysis />} />
          <Route path="/branch-analysis/:plantId" element={<BranchAnalysis />} />
        </Routes>
      </AnimatePresence>
    </div>
  );
}

export default App;