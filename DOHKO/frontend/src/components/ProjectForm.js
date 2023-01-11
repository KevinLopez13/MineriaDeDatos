import React, {useState} from 'react';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';
import FormControl from 'react-bootstrap/FormControl';
import Modal from 'react-bootstrap/Modal';
import axios from 'axios';
import { FiPlusCircle } from 'react-icons/fi';

export default function ProjectForm({projects, setProjects}) {

    const [show, setShow] = useState(false);

    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);
    // const [radioValue, setRadioValue] = useState(true);

    // API
    const [name, setName] = useState('');
    const [url, setUrl] = useState('');
    const [dataFile, setFile] = useState(null);
    const [desc, setDesc] = useState('');

    const onChangeName = e => {setName(e.target.value);}
    const onChangeUrl = e => {setUrl(e.target.value);}
    const onChangeFile = e =>{setFile(e.target.files[0]);}
    const onChangeDesc = e => {setDesc(e.target.value);}

    const handleSubmit = e => {
        e.preventDefault();
        if (!name | (!url & !dataFile) ){
            alert("Ingresa un nombre y una fuente de datos.")
            return;
        }

        axios.post('/post/',{
            name:name,
            url:url,
            dataFile:dataFile,
            desc:desc,
        },{
            headers: {
            "Content-Type": "multipart/form-data",
        }
        })

        .then((response) => {
            setName('');
            setUrl('');
            setFile(null);
            const {data} = response;
            setProjects([...projects, data])
            .catch(() => {
                alert('Algo malo pasó!')
            })
        })
        setShow(false);
    }
    
    return (
        <>
        <FiPlusCircle size={30} style={{cursor:'pointer'}} onClick={handleShow}/>
        <Modal show={show} onHide={handleClose}>
            <Modal.Header closeButton>
                <Modal.Title>Nuevo Proyecto</Modal.Title>
            </Modal.Header>
            
            <Modal.Body>
                <Form onSubmit={handleSubmit}>
                    <InputGroup className='mb-3'>
                        <InputGroup.Text>Nombre</InputGroup.Text>
                        <FormControl type="text" onChange={onChangeName}/>
                    </InputGroup>

                    <InputGroup className='mb-3'>
                        <InputGroup.Text>Descripción</InputGroup.Text>
                        <FormControl type="text" onChange={onChangeDesc}/>
                    </InputGroup>

                    <InputGroup className='mb-3'>
                        <FormControl type="file" accept='.csv' onChange={onChangeFile}/>
                    </InputGroup>
                    
                    <InputGroup className='mb-3'>
                        <FormControl type="text" placeholder='Introduce una URL' onChange={onChangeUrl} />
                    </InputGroup>
                    
                    <Button variant="dark" type='submit'>
                        Guardar
                    </Button>
                </Form>
            </Modal.Body>
        </Modal>
        </>
    )
}
