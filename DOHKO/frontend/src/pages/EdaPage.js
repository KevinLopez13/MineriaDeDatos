import React from 'react';
import Card from 'react-bootstrap/Card';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Console from '../components/Console';

export default function EdaPage({id=0}) {

    return (
        
    <Row className='justify-content-md-center' md={2}>
        <Col xs="auto" md={10}>

            <h4>Análisis Exploratorio de Datos</h4>

            <Row className='mt-4'>
                <Col>
                <Card>
                    <Card.Body>
                    <Card.Title><h5>Previsualización de datos</h5></Card.Title>
                        <Console
                            id_project={id}
                            url={'eda/dataPreview/'}
                        />
                    </Card.Body>
                </Card>
                </Col>
            </Row>

            <Row className='mt-4'>
                <Col>
                    <Card><Card.Body>
                    <Card.Title>1. Descripción de la estructura de datos</Card.Title>
                        <div className='mt-3'>
                            <h6>Dimensiones del dataframe</h6>
                            <Console
                                id_project={id}
                                url={'eda/dataShape/'}
                            />
                        </div>
                        
                        <div className='mt-3'>
                            <h6>Tipos de datos de variables</h6>
                            <Console
                                id_project={id}
                                url={'eda/dataTypes/'}
                            />
                        </div>
                    </Card.Body></Card>
                </Col>
            </Row>

            <Row className='mt-4'>

                <Col>
                    <Card><Card.Body>
                    <Card.Title>2. Identificación de datos faltantes</Card.Title>
                        
                        <div className='col mt-3'>
                            <h6>Datos nulos</h6>
                                <Console
                                    id_project={id}
                                    url={'eda/dataNull/'}
                                />
                        </div>
                        <div className='col mt-3'>
                            <h6>Eliminación de datos nulos</h6>
                            <Console
                                id_project={id}
                                url={'eda/dataDropna/'}
					        />
                        </div>
                    </Card.Body></Card>
                </Col>

            </Row>

            <Row className='mt-4'>
                <Col>
                    <Card><Card.Body>
                    <Card.Title>3. Detección de valores atípicos</Card.Title>

                    <div className='mt-3'>
                        <h6>Distribución de variables numéricas</h6>
                        <Console
                            id_project={id}
                            url={'eda/dataHistogram/'}
                        />
                    </div>

                    <div className='mt-3'>
                        <h6>Resumen estadístico de variables numéricas</h6>
                        <Console
                            id_project={id}
                            url={'eda/dataDescribe/'}
                        />
                    </div>

                    <div className='mt-3'>
                        <h6>Diagramas para detectar posibles valores atípicos</h6>
                        <Console
                            id_project={id}
                            url={'eda/dataBoxplot/'}
                        />
                    </div>

                    <div className='mt-3'>
                        <h6>Distribución de variables categóricas</h6>
                        <Console
                            id_project={id}
                            url={'eda/dataObjectBar/'}
                        />
                    </div>

                    </Card.Body></Card>
                </Col>
            </Row>

            <Row className='mt-4'>
                <Col>
                    <Card><Card.Body>
                    <Card.Title>4. Identificación de relaciones entre pares variables</Card.Title>
                    <div className='mt-3'>
                        <h6>Matriz de correlaciones</h6>
                        <Console
                            id_project={id}
                            url={'eda/dataCorrelation/'}
                        />
                    </div>
                    </Card.Body></Card>
                </Col>
            </Row>
        </Col>
    </Row>
  );
}