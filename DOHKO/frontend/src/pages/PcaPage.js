import React, { useState, useEffect } from 'react';
import Card from 'react-bootstrap/Card';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Code from '../components/Code';
import Console from '../components/Console';
import LineChart from '../components/LineChart';

export default function PcaPage({id=0}) {
	return (
		<Row className='justify-content-md-center' md={2}>
        <Col xs="auto" md={10}>

            <h4>Análisis de Componentes Principales</h4>

            <Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Previsualización de datos</h5></Card.Title>
					<Console
						id_project={id}
						url={'pca/dataPreview/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Evidencia de variables relacionadas</h5></Card.Title>
					<Console
						id_project={id}
						url={'pca/dataCorrelation/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Eliminación de datos nulos</h5></Card.Title>
					<Console
						id_project={id}
						url={'eda/dataDropna/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Estandarización de datos</h5></Card.Title>
					<Console
						id_project={id}
						url={'pca/dataScaler/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Matriz de covarianzas, componentes y varianza</h5></Card.Title>
					<Console
						id_project={id}
						url={'pca/dataComponents/'}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Número de componentes principales</h5></Card.Title>
					
					<div className='row'>
                        <div className='col'>
							<Code
								command={'varianza = pca.explained_variance_ratio_'}
							/>
						</div>
					</div>
					
					<div className='row'>
						<h6>Proporción de varianza</h6>
						<div className='col'>
							<Console
								id_project={id}
								url={'pca/dataVariance/'}
							/>
						</div>
                        <div className='col'>
							<Console
								id_project={id}
								url={'pca/dataVarianceAcumLine/'}
							/>
						</div>
					</div>

					<div className='row'>
						<div className='col my-3'>
							<h6>Varianza acumulada</h6>
							<Console
								id_project={id}
								url={'pca/dataVarianceAcum/'}
							/>
						</div>
					</div>
                </Card.Body></Card>
                </Col>
            </Row>

			<Row className='mt-4'>
                <Col>
                <Card><Card.Body>
				<Card.Title><h5>Proporción de relevancias</h5></Card.Title>
					<Console
						id_project={id}
						url={'pca/dataWeights/'}
					/>
					<Console
						id_project={id}
						url={'pca/dataDrop/'}
						update={true}
					/>
                </Card.Body></Card>
                </Col>
            </Row>

		</Col>
		</Row>
	);
}