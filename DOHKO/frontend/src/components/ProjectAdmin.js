import React, {useState} from 'react';
import Card from 'react-bootstrap/Card';
import {FaEdit, FaPlay, FaTrashAlt} from 'react-icons/fa';
import axios from 'axios';
import Col from 'react-bootstrap/esm/Col';

export default function ProjectAdmin({projects = [], setProjects}) {

    const handleDelete = (id) => {
        axios.delete(`/delete/${id}`)
        .then(() => {
            const newProjects = projects.filter(project => {
                return project.id !== id
            });
            setProjects(newProjects);
        }).catch(() => {
            alert('kk')
        })
    }

    return (
        <div className='row row-cols-1 row-cols-md-4 g-4'>
            {projects.map( project => {
                return (
                    <Col>
                    <Card className="text-center h-100">
                    <Card.Header>
                    <Card.Title>{project.name}</Card.Title>
                    </Card.Header>
                    <Card.Body>
                        <Card.Text>
                        <p>{project.desc}</p>
                        <p>{project.fileName}</p>
                        </Card.Text>
                    </Card.Body>
                    <Card.Footer>
                        <FaTrashAlt
                            className="mx-3"
                            onClick={()=>{handleDelete(project.id)}}
                            size={20}
                            style={{cursor:'pointer'}}
                        />
                    </Card.Footer>
                    </Card>
                    </Col>
                )
            })}
        </div>
    )
}
