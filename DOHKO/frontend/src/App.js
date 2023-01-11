import React, { useState } from "react";

import Header from "./components/Header";
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';

import EdaPage from "./pages/EdaPage";
import PcaPage from "./pages/PcaPage";
import ProjectPage from "./pages/ProjectPage";
import PronosticPage from "./pages/PronosticPage";
import ClasificationPage from "./pages/ClasificationPage";
import ClusteringPage from "./pages/ClusteringPage";
import SVMPage from "./pages/SVMPage";

function App() {

  const [idP, setId] = useState(0);

  return (
    <>
        <Router>
            <Routes>
                    <Route path="/" element={<Header id={idP} setId={setId} />}>
                    <Route exac path="/eda" element={<EdaPage id={idP} />} />
                    <Route exac path="/pca" element={<PcaPage id={idP} />} />
                    <Route exac path="/projects" element={<ProjectPage id={idP} />} />
                    <Route exac path="/pronostic" element={<PronosticPage id={idP} />} />
                    <Route exac path="/clasification" element={<ClasificationPage id={idP} />} />
                    <Route exac path="/cluster" element={<ClusteringPage id={idP} />} />
                    <Route exac path="/svm" element={<SVMPage id={idP} />} />
                </Route>
            </Routes>
        </Router>
    </>
  );
}

export default App;
