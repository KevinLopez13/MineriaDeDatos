import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import {Outlet, Link} from 'react-router-dom';
import axios from 'axios';
import React, { useState, useEffect } from 'react';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import Dropdown from 'react-bootstrap/Dropdown';

export default function Header({id, setId}) {

  const [nameProject, setNameProject] = useState("Proyecto");
  const [projects, setProjects] = useState([]);

    useEffect(() => {
        axios.get('/get/')
        .then((response) => {
            setProjects(response.data)
        }).catch(() => {
            alert('KK')
        })
    },[id])


  const setIdProject = (id) => {
    const currentProject = projects.filter(project => {
      return project.id === parseInt(id, 10)
    });
    setId(id);
    setNameProject(currentProject[0].name);
  }

    

  return (
    <>
    <Navbar expand="lg" bg="dark" variant="dark" sticky="top" >
      <Container>
        <Navbar.Brand as={Link} to="/">
            <img
              alt=""
              src="../../favicon.ico"
              width="30"
              height="30"
              className="d-inline-block align-top"
            />{' '}
            DOHKO</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">

          <Nav className="me-auto">
            
            <Nav.Link as={Link} to="/projects">Proyectos</Nav.Link>
            
            <Dropdown as={ButtonGroup} onSelect={(id) => {setIdProject(id)}} >
              <Dropdown.Toggle variant="dark" id="dropdown" >{`${nameProject}`}</Dropdown.Toggle>
              <Dropdown.Menu variant="dark">
                {projects.map((project, index)=>{
                  return (
                  <Dropdown.Item eventKey={project.id}>{`${project.name}`}</Dropdown.Item>
                  )
                })}
              </Dropdown.Menu>
            </Dropdown>
            
            <Nav.Link as={Link} to="/eda">EDA</Nav.Link>
            <Nav.Link as={Link} to="/pca">PCA</Nav.Link>
            <Nav.Link as={Link} to="/pronostic">Pronóstico</Nav.Link>
            <Nav.Link as={Link} to="/clasification">Clasificación</Nav.Link>
            <Nav.Link as={Link} to="/cluster">Híbrido</Nav.Link>
            <Nav.Link as={Link} to="/svm">SVM</Nav.Link>

          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
    <div className="container-fluid p-4">
        <Outlet>
        </Outlet>
    </div>
    </>
  );
}