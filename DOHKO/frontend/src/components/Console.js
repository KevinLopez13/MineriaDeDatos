import React, { useState, useEffect } from 'react';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';
import axios from 'axios';
import { FaPlay } from 'react-icons/fa';
import { TfiReload } from "react-icons/tfi";
import DataFrame from './DataFrame';
import HeatMap from './HeatMap';
import Bar from './Bar';
import BoxPlot from './BoxPlot';
import CheckBox from './CheckBox';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import LineChart from './LineChart';
import Spinner from 'react-bootstrap/Spinner';
import Figure from 'react-bootstrap/Figure';
import Inputs from './Inputs';

export default function Console({id_project, url, update=false}) {

    // GET
    const [data, setData] = useState([]);
    const [command, setCommand] = useState('');
    const [args, setArgs] = useState([]);
    const [kargs, setKargs] = useState([]);
    const [vars, setVars] = useState([]);

    // Default arguments
    const [args_def, setArgs_def] = useState([]);
    // Default arguments with user values
    const [ input_vals, setInputVals] = useState({});
    // Image
    const [image, setImage] = useState("");
    // POST
    const [usrVars, setUsrVars] = useState([]);
    // CONTROL
    const [loading, setLoading] = useState(false);

    
    const getfun = () => {
        setLoading(true);
        axios.get(`${url}${id_project}`)
        .then((response) => {
            setData(response.data);
            setLoading(false);
        }).catch(() => {
            setLoading(false);
            alert('Algo salió mal!');
        })
    }

    const postfun = () => {
        setLoading(true);
        axios.post(`${url}${id_project}/`, { 'kargs':args, 'usrVars':usrVars, 'input_vals': input_vals })
        .then((response) => {
            setData(response.data);
            setLoading(false);
        }).catch(() => {
            setLoading(false);
            alert('Algo salió mal!');
        })
    }

    const config = () => {
        axios.post(`dhk/getParams/${id_project}/`,{'func':url})
            .then((response) => {
                setCommand(response.data.command);
                setArgs_def(response.data.default_args);
                setVars(response.data.vars);
                setData(response.data);
                setImage(response.data.imageROC);
                console.log(response.data);
            }).catch(() => {
                alert('Error al obtener la configuración de la consola!')
            })
    }

    useEffect(() => {
        if ( id_project !== 0 && !update ){
            config();
        }},[id_project]
        )


    return (
        <div className='mt-3'>
            <Row className='my-3'>
                <Col>
                <InputGroup size="sm">
                    { update ? 
                        <InputGroup.Text>
                            <TfiReload
                                size={15}
                                style={{cursor:'pointer'}}
                                onClick={() =>{ config() } }
                            />
                        </InputGroup.Text> : null }

                    <InputGroup.Text>
                        {loading === false ?
                            <FaPlay 
                                size={15}
                                style={{cursor:'pointer'}}
                                onClick={() =>{ kargs || vars ? postfun() : getfun() } }
                            />
                            :
                            <Spinner animation="border" variant="dark" size="sm" />
                        }
                    </InputGroup.Text>

                    <Form.Control
                        as={ data.multiline ? "textarea" : "input"}
                        style={{ fontFamily:"monospace"}}
                        defaultValue={command}
                        readOnly
                    />

                </InputGroup>
                </Col>
            </Row>

            <Row className='mt-3'>

                { vars ?
                    <Col className='mb-3' md={3}>
                        <CheckBox data={vars} type={data.checkBoxType} vars={usrVars} setVars={setUsrVars} />
                    </Col> : null }
                
                { args_def ? 
                    <Col className='mb-3' md={3}>
                        <Inputs vals={args_def} input_vals={input_vals} setInputVals={setInputVals}/>
                    </Col> : null }
                
                <Col md={ vars || args_def ? 9 : 12 }>
                    {data.resType === 'table' ? <div> <DataFrame data={data}/> </div> : null}
                    {data.complement === 'heatmap' ? <HeatMap data={data}/> : null}
                    {data.resType === 'text' ? <div className='mb-3' style={{fontSize:14}}>{data.code.result}</div> : null}
                    {data.resType === 'boxp' ? <div><BoxPlot boxes={data.dataGraph}/></div> : null}
                    {data.resType === 'hist' ? <Bar hists={data.dataGraph} /> : null}
                    {data.resType === 'line' ? <div className='mt-3'><LineChart line={data.dataGraph}/></div> : null}
                </Col>
                
                {data.resType === 'img' ?
                    <div style={{maxHeight:"350px", overflow:"auto", textAlign:"center"}}>
                        <Figure.Image
                            src={data.images}
                        />
                    </div> : null
                }
            </Row>
        </div>
    )
}