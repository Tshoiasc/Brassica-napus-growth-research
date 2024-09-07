// src/SecondPage.js

import React from 'react';
import { Button } from 'antd';
import { Link } from 'react-router-dom';
import withPageTransition from './withPageTransition';

function SecondPage() {
  return (
    <div>
      <h1>Second Page</h1>
      <Link to="/">
        <Button>Back to Home</Button>
      </Link>
    </div>
  );
}

export default withPageTransition(SecondPage);