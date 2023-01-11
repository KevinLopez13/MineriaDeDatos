import React, { useState, useEffect } from 'react';
import axios from 'axios';

import ProjectAdmin from "../components/ProjectAdmin";
import ProjectForm from '../components/ProjectForm';

export default function ProjectPage() {

    const [project, setProjects] = useState([]);

    useEffect(() => {
        axios.get('/get/')
        .then((response) => {
            setProjects(response.data)
        }).catch(() => {
            alert('KK')
        })
    },[])

    return (
        <>
        <h4>Proyectos</h4>
        <div className='my-3'>
            <ProjectForm 
            projects={project}
            setProjects={setProjects}
            />
        </div>
            <ProjectAdmin
            projects={project}
            setProjects={setProjects} 
            />
        </>
    )
}