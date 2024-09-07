// src/withPageTransition.js

import React from 'react';
import { motion } from 'framer-motion';
import { pageVariants, pageTransition } from './animationConfig';

const withPageTransition = (WrappedComponent) => {

  return function WithPageTransition(props) {
    // useEffect(() => {
    //     // 禁用滚动
    //     document.body.style.overflow = 'hidden';
        
    //     return () => {
    //       // 组件卸载时恢复滚动
    //       document.body.style.overflow = 'unset';
    //     };
    //   }, []);
    return (
      <motion.div
        initial="initial"
        animate="in"
        exit="out"
        variants={pageVariants}
        transition={pageTransition}
      >
        <WrappedComponent {...props} />
      </motion.div>
    );
  };
};

export default withPageTransition;