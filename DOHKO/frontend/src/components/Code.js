import React from 'react';
import { FaPython } from 'react-icons/fa';
import InputGroup from 'react-bootstrap/InputGroup';
import Form from 'react-bootstrap/Form';


export default function Code({command}) {

    return (
        <div className='my-3'>
            <InputGroup size="sm">
                <InputGroup.Text>
                    <FaPython size={15}/>
                </InputGroup.Text>
                <Form.Control
                    as={command.length < 128 ? "input" : "textarea"}
                    style={{ fontFamily:"monospace"}}
                    defaultValue={command}
                    readOnly
                />
            </InputGroup>
        </div>
    )
}